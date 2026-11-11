
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import csv
import time


# ============================================================
# 1. PATHS AND PARAMETERS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_1_PATH = os.path.join(
    BASE_DIR, "MPI-Sintel-training_images",
    "training", "clean", "alley_1", "frame_0001.png"
)

IMAGE_2_PATH = os.path.join(
    BASE_DIR, "MPI-Sintel-training_images",
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
    "results", "Task 4"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD IMAGES
# ============================================================

image1 = cv2.imread(IMAGE_1_PATH, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(IMAGE_2_PATH, cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None:
    raise FileNotFoundError("Could not load Sintel images.")

if image1.shape != image2.shape:
    raise ValueError("Input images must have identical dimensions.")

height, width = image1.shape

print("=" * 65)
print("STAGE 2 - TASK 4: DIS DENSE OPTICAL FLOW")
print("=" * 65)

print(f"Image dimensions: {width} x {height}")


# ============================================================
# 3. LOAD GROUND-TRUTH FLOW
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

invalid_mask = cv2.imread(
    INVALID_MASK_PATH,
    cv2.IMREAD_GRAYSCALE
)

if invalid_mask is None:
    raise FileNotFoundError("Could not load Sintel invalid mask.")

valid_mask = invalid_mask == 0

gt_u = ground_truth[:, :, 0]
gt_v = ground_truth[:, :, 1]


# ============================================================
# 4. DEFINE DIS PRESETS
# ============================================================

presets = {
    "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
}

results = {}


# ============================================================
# 5. CALCULATE DIS OPTICAL FLOW
# ============================================================

for preset_name, preset_value in presets.items():

    print("\n" + "-" * 50)
    print(f"Running DIS preset: {preset_name}")

    dis = cv2.DISOpticalFlow_create(preset_value)

    # Use the same default DIS settings for all three presets.
    # Keep spatial propagation and variational refinement enabled.
    dis.setUseSpatialPropagation(True)

# Enable variational refinement
    dis.setVariationalRefinementIterations(5)
    dis.setVariationalRefinementAlpha(20.0)
    dis.setVariationalRefinementDelta(5.0)
    dis.setVariationalRefinementGamma(10.0)

    start_time = time.perf_counter()

    flow = dis.calc(image1, image2, None)

    elapsed_time = time.perf_counter() - start_time

    if flow is None or flow.shape != (height, width, 2):
        raise RuntimeError(
            f"DIS {preset_name} returned an invalid flow array."
        )

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    magnitude, direction = cv2.cartToPolar(
        u, v, angleInDegrees=True
    )

    # ========================================================
    # 6. EVALUATE AGAINST GROUND TRUTH
    # ========================================================

    epe = np.sqrt(
        (u - gt_u) ** 2 +
        (v - gt_v) ** 2
    )

    dot_product = u * gt_u + v * gt_v + 1.0

    denominator = np.sqrt(
        u ** 2 + v ** 2 + 1.0
    ) * np.sqrt(
        gt_u ** 2 + gt_v ** 2 + 1.0
    )

    cosine = np.clip(
        dot_product / denominator,
        -1.0,
        1.0
    )

    angular_error = np.degrees(np.arccos(cosine))

    valid_epe = epe[valid_mask]
    valid_angular_error = angular_error[valid_mask]

    metrics = {
        "valid_pixels": int(valid_mask.sum()),
        "mean_epe": float(np.mean(valid_epe)),
        "median_epe": float(np.median(valid_epe)),
        "std_epe": float(np.std(valid_epe)),
        "max_epe": float(np.max(valid_epe)),
        "epe_below_1": float(np.mean(valid_epe < 1.0) * 100),
        "epe_below_3": float(np.mean(valid_epe < 3.0) * 100),
        "epe_below_5": float(np.mean(valid_epe < 5.0) * 100),
        "mean_angular_error": float(
            np.mean(valid_angular_error)
        ),
        "runtime_seconds": float(elapsed_time)
    }

    results[preset_name] = {
        "flow": flow,
        "magnitude": magnitude,
        "direction": direction,
        "epe": epe,
        "metrics": metrics
    }

    print(f"Runtime: {elapsed_time:.4f} seconds")
    print(f"Mean EPE: {metrics['mean_epe']:.6f} px")
    print(f"Median EPE: {metrics['median_epe']:.6f} px")
    print(
        f"Mean angular error: "
        f"{metrics['mean_angular_error']:.6f} degrees"
    )


# ============================================================
# 7. SAVE VISUALIZATIONS FOR EACH PRESET
# ============================================================

for preset_name, result in results.items():

    flow = result["flow"]
    magnitude = result["magnitude"]
    direction = result["direction"]
    epe = result["epe"]

    # Flow color visualization
    hsv = np.zeros((height, width, 3), dtype=np.uint8)

    hsv[:, :, 0] = np.uint8(direction / 2)
    hsv[:, :, 1] = 255

    normalized_magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    hsv[:, :, 2] = normalized_magnitude.astype(np.uint8)

    flow_color = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            f"dis_{preset_name.lower()}_flow.png"
        ),
        flow_color
    )

    # Save NumPy arrays
    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"dis_{preset_name.lower()}_flow.npy"
        ),
        flow
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"dis_{preset_name.lower()}_magnitude.npy"
        ),
        magnitude
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"dis_{preset_name.lower()}_epe.npy"
        ),
        epe
    )

    # Endpoint error map
    plt.figure(figsize=(11, 5))

    plt.imshow(
        np.where(valid_mask, epe, np.nan),
        cmap="jet"
    )

    plt.colorbar(label="Endpoint error (pixels)")
    plt.title(f"DIS {preset_name} - Endpoint Error")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"dis_{preset_name.lower()}_epe.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. CREATE PRESET ACCURACY COMPARISON
# ============================================================

preset_names = list(results.keys())

mean_epe_values = [
    results[name]["metrics"]["mean_epe"]
    for name in preset_names
]

runtime_values = [
    results[name]["metrics"]["runtime_seconds"]
    for name in preset_names
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(preset_names, mean_epe_values)
axes[0].set_title("DIS Preset - Mean EPE")
axes[0].set_ylabel("Mean EPE (pixels)")
axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(preset_names, runtime_values)
axes[1].set_title("DIS Preset - Runtime")
axes[1].set_ylabel("Runtime (seconds)")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "dis_preset_comparison.png"),
    dpi=200
)

plt.close()


# ============================================================
# 9. CREATE EPE DISTRIBUTION COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

for preset_name in preset_names:

    plt.hist(
        results[preset_name]["epe"][valid_mask],
        bins=60,
        alpha=0.4,
        label=preset_name
    )

plt.xlabel("Endpoint Error (pixels)")
plt.ylabel("Number of pixels")
plt.title("DIS Preset - Endpoint Error Distribution")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "dis_epe_distribution.png"),
    dpi=200
)

plt.close()


# ============================================================
# 10. SAVE CSV METRICS
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "dis_metrics.csv"
)

with open(csv_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "preset",
        "mean_epe",
        "median_epe",
        "std_epe",
        "max_epe",
        "mean_angular_error",
        "epe_below_1_percent",
        "epe_below_3_percent",
        "epe_below_5_percent",
        "runtime_seconds"
    ])

    for preset_name in preset_names:

        metrics = results[preset_name]["metrics"]

        writer.writerow([
            preset_name,
            metrics["mean_epe"],
            metrics["median_epe"],
            metrics["std_epe"],
            metrics["max_epe"],
            metrics["mean_angular_error"],
            metrics["epe_below_1"],
            metrics["epe_below_3"],
            metrics["epe_below_5"],
            metrics["runtime_seconds"]
        ])


# ============================================================
# 11. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "dis_summary.txt"
)

with open(summary_path, "w") as file:

    file.write("STAGE 2 - TASK 4\n")
    file.write("DIS DENSE OPTICAL FLOW\n\n")

    file.write("Dataset: MPI Sintel\n")
    file.write("Sequence: alley_1\n")
    file.write("Frames: frame_0001.png -> frame_0002.png\n")
    file.write(f"Image dimensions: {width} x {height}\n\n")

    file.write("Evaluation uses valid Sintel pixels.\n\n")

    for preset_name in preset_names:

        metrics = results[preset_name]["metrics"]

        file.write("=" * 45 + "\n")
        file.write(f"DIS preset: {preset_name}\n")

        for name, value in metrics.items():
            file.write(f"{name}: {value}\n")

        file.write("\n")


# ============================================================
# 12. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("DIS PRESET COMPARISON")
print("=" * 65)

for preset_name in preset_names:

    metrics = results[preset_name]["metrics"]

    print(f"\n{preset_name}")

    for name, value in metrics.items():
        print(f"{name}: {value}")

print("\nTask 4 completed successfully.")
print(f"Results saved to: {OUTPUT_DIR}")
