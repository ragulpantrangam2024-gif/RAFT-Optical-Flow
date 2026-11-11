
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import struct


# ============================================================
# 1. PATHS AND PARAMETERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_1_PATH = os.path.join(
    BASE_DIR,
    "MPI-Sintel-training_images",
    "training", "clean", "alley_1", "frame_0001.png"
)

IMAGE_2_PATH = os.path.join(
    BASE_DIR,
    "MPI-Sintel-training_images",
    "training", "clean", "alley_1", "frame_0002.png"
)

GT_FLOW_PATH = os.path.join(
    BASE_DIR, "training", "flow", "alley_1", "frame_0001.flo"
)

INVALID_MASK_PATH = os.path.join(
    BASE_DIR, "training", "invalid", "alley_1", "frame_0001.png"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results", "Task 2"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CORNERS = 300
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7

WINDOW_SIZE = (21, 21)
PYRAMID_LEVELS = [0, 3]

CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01
)


# ============================================================
# 2. LOAD IMAGES
# ============================================================

image1 = cv2.imread(IMAGE_1_PATH, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(IMAGE_2_PATH, cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None:
    raise FileNotFoundError("Could not load Sintel input images.")

if image1.shape != image2.shape:
    raise ValueError("Input images must have identical dimensions.")

height, width = image1.shape

print("=" * 60)
print("STAGE 2 - TASK 2: PYRAMIDAL LUCAS-KANADE")
print("=" * 60)
print(f"Image dimensions: {width} x {height}")


# ============================================================
# 3. LOAD SINTEL GROUND-TRUTH FLOW
# ============================================================

def read_flo_file(path):
    with open(path, "rb") as file:
        magic = struct.unpack("f", file.read(4))[0]

        if magic != 202021.25:
            raise ValueError("Invalid .flo file.")

        flow_width = struct.unpack("i", file.read(4))[0]
        flow_height = struct.unpack("i", file.read(4))[0]

        data = np.fromfile(
            file,
            dtype=np.float32,
            count=flow_width * flow_height * 2
        )

    return data.reshape(flow_height, flow_width, 2)


ground_truth = read_flo_file(GT_FLOW_PATH)

if ground_truth.shape[:2] != image1.shape:
    raise ValueError("Ground-truth dimensions do not match the images.")

invalid_mask = cv2.imread(INVALID_MASK_PATH, cv2.IMREAD_GRAYSCALE)

if invalid_mask is None:
    raise FileNotFoundError("Could not load Sintel invalid mask.")


# ============================================================
# 4. DETECT FEATURE POINTS
# ============================================================

points = cv2.goodFeaturesToTrack(
    image1,
    maxCorners=MAX_CORNERS,
    qualityLevel=QUALITY_LEVEL,
    minDistance=MIN_DISTANCE,
    blockSize=7
)

if points is None:
    raise RuntimeError("No feature points detected.")

print(f"Detected feature points: {len(points)}")


# ============================================================
# 5. RUN LUCAS-KANADE AT DIFFERENT PYRAMID LEVELS
# ============================================================

results = {}

for level in PYRAMID_LEVELS:

    print(f"\nRunning Lucas-Kanade with maxLevel={level}...")

    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        image1,
        image2,
        points,
        None,
        winSize=WINDOW_SIZE,
        maxLevel=level,
        criteria=CRITERIA
    )

    if next_points is None or status is None:
        raise RuntimeError(f"Tracking failed at pyramid level {level}.")

    status = status.ravel().astype(bool)

    old_points = points.reshape(-1, 2)
    new_points = next_points.reshape(-1, 2)

    flow_vectors = new_points - old_points
    magnitudes = np.linalg.norm(flow_vectors, axis=1)

    results[level] = {
        "new_points": new_points,
        "status": status,
        "flow": flow_vectors,
        "magnitudes": magnitudes
    }

    print(f"Successfully tracked: {np.count_nonzero(status)}")
    print(f"Tracking failures: {np.count_nonzero(~status)}")


# ============================================================
# 6. SAMPLE GROUND TRUTH AT FEATURE LOCATIONS
# ============================================================

def evaluate_tracking(result):

    old_points = points.reshape(-1, 2)
    new_points = result["new_points"]
    status = result["status"]
    estimated_flow = result["flow"]

    # Use nearest-pixel mask lookup at the original feature location.
    x = np.rint(old_points[:, 0]).astype(int)
    y = np.rint(old_points[:, 1]).astype(int)

    in_bounds = (
        (x >= 0) & (x < width) &
        (y >= 0) & (y < height)
    )

    valid_mask = np.zeros(len(points), dtype=bool)
    valid_mask[in_bounds] = invalid_mask[
        y[in_bounds], x[in_bounds]
    ] == 0

    valid_mask &= status

    indices = np.where(valid_mask)[0]

    if len(indices) == 0:
        raise RuntimeError("No valid tracked points for evaluation.")

    # Bilinear sampling of ground-truth flow at original point locations.
    map_x = old_points[indices, 0].astype(np.float32).reshape(-1, 1)
    map_y = old_points[indices, 1].astype(np.float32).reshape(-1, 1)

    gt_u = cv2.remap(
        ground_truth[:, :, 0],
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR
    ).ravel()

    gt_v = cv2.remap(
        ground_truth[:, :, 1],
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR
    ).ravel()

    gt_flow = np.column_stack((gt_u, gt_v))
    predicted_flow = estimated_flow[indices]

    endpoint_error = np.linalg.norm(
        predicted_flow - gt_flow,
        axis=1
    )

    angular_error = np.degrees(
        np.arccos(
            np.clip(
                (np.sum(predicted_flow * gt_flow, axis=1) + 1.0) /
                (
                    np.sqrt(np.sum(predicted_flow ** 2, axis=1) + 1.0) *
                    np.sqrt(np.sum(gt_flow ** 2, axis=1) + 1.0)
                ),
                -1.0,
                1.0
            )
        )
    )

    return {
        "valid_indices": indices,
        "epe": endpoint_error,
        "angular_error": angular_error,
        "mean_epe": float(np.mean(endpoint_error)),
        "median_epe": float(np.median(endpoint_error)),
        "mean_angular_error": float(np.mean(angular_error)),
        "below_1": float(np.mean(endpoint_error < 1.0) * 100),
        "below_3": float(np.mean(endpoint_error < 3.0) * 100),
        "below_5": float(np.mean(endpoint_error < 5.0) * 100)
    }


evaluations = {}

for level in PYRAMID_LEVELS:
    evaluations[level] = evaluate_tracking(results[level])


# ============================================================
# 7. SAVE TRACKING VISUALIZATIONS
# ============================================================

old_points = points.reshape(-1, 2)

for level in PYRAMID_LEVELS:

    result = results[level]
    tracked_image = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    for index, (old, new) in enumerate(
        zip(old_points, result["new_points"])
    ):
        if result["status"][index]:
            p1 = tuple(np.rint(old).astype(int))
            p2 = tuple(np.rint(new).astype(int))

            cv2.circle(tracked_image, p2, 2, (0, 255, 0), -1)
            cv2.line(tracked_image, p1, p2, (0, 0, 255), 1)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"tracking_maxLevel_{level}.png"
    )

    cv2.imwrite(output_path, tracked_image)


# ============================================================
# 8. CREATE COMPARISON VISUALIZATION
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, level in zip(axes, PYRAMID_LEVELS):

    result = results[level]
    evaluation = evaluations[level]

    ax.imshow(image2, cmap="gray")

    valid_indices = np.where(result["status"])[0]

    ax.quiver(
        old_points[valid_indices, 0],
        old_points[valid_indices, 1],
        result["flow"][valid_indices, 0],
        result["flow"][valid_indices, 1],
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1
    )

    ax.set_title(
        f"maxLevel={level}\n"
        f"Tracked: {len(valid_indices)} | "
        f"Mean EPE: {evaluation['mean_epe']:.4f} px"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "lk_pyramid_comparison.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. SAVE EPE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

for level in PYRAMID_LEVELS:
    plt.hist(
        evaluations[level]["epe"],
        bins=30,
        alpha=0.5,
        label=f"maxLevel={level}"
    )

plt.xlabel("Endpoint Error (pixels)")
plt.ylabel("Number of feature points")
plt.title("Single-Level vs Pyramidal Lucas-Kanade")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "lk_epe_distribution.png"),
    dpi=200
)

plt.close()


# ============================================================
# 10. SAVE METRICS CSV
# ============================================================

csv_path = os.path.join(OUTPUT_DIR, "lk_pyramid_metrics.csv")

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "maxLevel",
        "detected_points",
        "tracked_points",
        "failed_points",
        "valid_evaluated_points",
        "mean_epe",
        "median_epe",
        "mean_angular_error_degrees",
        "epe_below_1px_percent",
        "epe_below_3px_percent",
        "epe_below_5px_percent"
    ])

    for level in PYRAMID_LEVELS:

        result = results[level]
        evaluation = evaluations[level]

        writer.writerow([
            level,
            len(points),
            int(np.count_nonzero(result["status"])),
            int(np.count_nonzero(~result["status"])),
            len(evaluation["valid_indices"]),
            evaluation["mean_epe"],
            evaluation["median_epe"],
            evaluation["mean_angular_error"],
            evaluation["below_1"],
            evaluation["below_3"],
            evaluation["below_5"]
        ])


# ============================================================
# 11. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "lk_pyramid_summary.txt"
)

with open(summary_path, "w") as file:

    file.write("STAGE 2 - TASK 2\n")
    file.write("PYRAMIDAL LUCAS-KANADE OPTICAL FLOW\n\n")

    file.write("Dataset: MPI Sintel\n")
    file.write("Sequence: alley_1\n")
    file.write("Frames: frame_0001.png -> frame_0002.png\n")
    file.write(f"Image dimensions: {width} x {height}\n\n")

    file.write(f"Detected feature points: {len(points)}\n")
    file.write(f"Window size: {WINDOW_SIZE}\n\n")

    for level in PYRAMID_LEVELS:

        result = results[level]
        evaluation = evaluations[level]

        file.write("-" * 45 + "\n")
        file.write(f"Lucas-Kanade maxLevel: {level}\n")
        file.write(
            f"Tracked points: {np.count_nonzero(result['status'])}\n"
        )
        file.write(
            f"Failed points: {np.count_nonzero(~result['status'])}\n"
        )
        file.write(
            f"Valid evaluated points: {len(evaluation['valid_indices'])}\n"
        )
        file.write(f"Mean EPE: {evaluation['mean_epe']:.6f} px\n")
        file.write(f"Median EPE: {evaluation['median_epe']:.6f} px\n")
        file.write(
            f"Mean angular error: "
            f"{evaluation['mean_angular_error']:.6f} degrees\n"
        )
        file.write(f"EPE below 1 px: {evaluation['below_1']:.2f}%\n")
        file.write(f"EPE below 3 px: {evaluation['below_3']:.2f}%\n")
        file.write(f"EPE below 5 px: {evaluation['below_5']:.2f}%\n\n")

    file.write(
        "Note: Evaluation is performed at detected feature locations, "
        "not over every image pixel.\n"
    )


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LUCAS-KANADE COMPARISON")
print("=" * 60)

for level in PYRAMID_LEVELS:

    result = results[level]
    evaluation = evaluations[level]

    print(f"\nmaxLevel = {level}")
    print(f"Tracked points: {np.count_nonzero(result['status'])}")
    print(f"Mean EPE: {evaluation['mean_epe']:.6f} px")
    print(f"Median EPE: {evaluation['median_epe']:.6f} px")
    print(
        f"Mean angular error: "
        f"{evaluation['mean_angular_error']:.6f} degrees"
    )
    print(f"EPE below 1 px: {evaluation['below_1']:.2f}%")
    print(f"EPE below 3 px: {evaluation['below_3']:.2f}%")
    print(f"EPE below 5 px: {evaluation['below_5']:.2f}%")


print("\nTask 2 completed successfully.")
print(f"Results saved to: {OUTPUT_DIR}")
