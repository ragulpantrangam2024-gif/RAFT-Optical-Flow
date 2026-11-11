
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import csv


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
    "results", "Task 3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD INPUT IMAGES
# ============================================================

image1 = cv2.imread(IMAGE_1_PATH, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(IMAGE_2_PATH, cv2.IMREAD_GRAYSCALE)

if image1 is None or image2 is None:
    raise FileNotFoundError("Could not load the Sintel input images.")

if image1.shape != image2.shape:
    raise ValueError("Input images must have identical dimensions.")

height, width = image1.shape

print("=" * 60)
print("STAGE 2 - TASK 3: TV-L1 DENSE OPTICAL FLOW")
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
    raise FileNotFoundError("Could not load the Sintel invalid mask.")

# Same mask convention used in the earlier Sintel evaluation tasks.
valid_mask = (invalid_mask == 0)


# ============================================================
# 4. CREATE TV-L1 OPTICAL FLOW ALGORITHM
# ============================================================

if not hasattr(cv2, "optflow"):
    raise RuntimeError(
        "OpenCV contrib module is missing. "
        "Install opencv-contrib-python in your virtual environment."
    )

if hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

elif hasattr(cv2.optflow, "createOptFlow_DualTVL1"):
    tvl1 = cv2.optflow.createOptFlow_DualTVL1()

else:
    raise RuntimeError("TV-L1 implementation is unavailable in this OpenCV build.")


# ============================================================
# 5. CONFIGURE TV-L1 PARAMETERS
# ============================================================

parameters = {
    "tau": 0.25,
    "lambda": 0.15,
    "theta": 0.3,
    "scales_number": 5,
    "warps_number": 5,
    "epsilon": 0.01,
    "inner_iterations": 30,
    "outer_iterations": 10,
    "scale_step": 0.8,
    "gamma": 0.0,
    "median_filtering": 5
}

setters = {
    "tau": "setTau",
    "lambda": "setLambda",
    "theta": "setTheta",
    "scales_number": "setScalesNumber",
    "warps_number": "setWarpingsNumber",
    "epsilon": "setEpsilon",
    "inner_iterations": "setInnerIterations",
    "outer_iterations": "setOuterIterations",
    "scale_step": "setScaleStep",
    "gamma": "setGamma",
    "median_filtering": "setMedianFiltering"
}

for name, value in parameters.items():

    setter_name = setters[name]

    if hasattr(tvl1, setter_name):
        getattr(tvl1, setter_name)(value)
    else:
        print(f"Warning: {setter_name} is unavailable in this OpenCV build.")


# ============================================================
# 6. CALCULATE DENSE OPTICAL FLOW
# ============================================================

print("\nCalculating TV-L1 optical flow...")
print("Please wait; dense optimization may take some time.")

flow = tvl1.calc(image1, image2, None)

if flow is None or flow.shape != (height, width, 2):
    raise RuntimeError("TV-L1 returned an invalid flow array.")

u = flow[:, :, 0]
v = flow[:, :, 1]

magnitude, direction = cv2.cartToPolar(u, v, angleInDegrees=True)

print("TV-L1 optical flow calculation completed.")


# ============================================================
# 7. CALCULATE EVALUATION METRICS
# ============================================================

gt_u = ground_truth[:, :, 0]
gt_v = ground_truth[:, :, 1]

epe = np.sqrt(
    (u - gt_u) ** 2 +
    (v - gt_v) ** 2
)

gt_magnitude = np.sqrt(gt_u ** 2 + gt_v ** 2)

# Sintel angular error convention, with an additional unit component.
dot_product = u * gt_u + v * gt_v + 1.0

denominator = np.sqrt(u ** 2 + v ** 2 + 1.0) * np.sqrt(
    gt_u ** 2 + gt_v ** 2 + 1.0
)

cosine = np.clip(dot_product / denominator, -1.0, 1.0)

angular_error = np.degrees(np.arccos(cosine))

valid_epe = epe[valid_mask]
valid_angular_error = angular_error[valid_mask]

if valid_epe.size == 0:
    raise RuntimeError("No valid ground-truth pixels available.")

metrics = {
    "valid_pixels": int(valid_mask.sum()),
    "mean_epe": float(np.mean(valid_epe)),
    "median_epe": float(np.median(valid_epe)),
    "std_epe": float(np.std(valid_epe)),
    "min_epe": float(np.min(valid_epe)),
    "max_epe": float(np.max(valid_epe)),
    "epe_below_1": float(np.mean(valid_epe < 1.0) * 100),
    "epe_below_3": float(np.mean(valid_epe < 3.0) * 100),
    "epe_below_5": float(np.mean(valid_epe < 5.0) * 100),
    "mean_angular_error": float(np.mean(valid_angular_error))
}


# ============================================================
# 8. CREATE FLOW COLOR VISUALIZATION
# ============================================================

hsv = np.zeros((height, width, 3), dtype=np.uint8)

hsv[:, :, 0] = np.uint8(direction / 2)
hsv[:, :, 1] = 255

# Normalize magnitude for visualization only.
normalized_magnitude = cv2.normalize(
    magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

hsv[:, :, 2] = normalized_magnitude.astype(np.uint8)

flow_visualization = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2BGR
)

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "tvl1_dense_optical_flow.png"),
    flow_visualization
)


# ============================================================
# 9. CREATE MOTION MAGNITUDE VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 5))

plt.imshow(magnitude, cmap="jet")
plt.colorbar(label="Motion magnitude (pixels)")
plt.title("TV-L1 Optical Flow - Motion Magnitude")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "tvl1_motion_magnitude.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. CREATE ENDPOINT ERROR MAP
# ============================================================

epe_visualization = np.where(valid_mask, epe, np.nan)

plt.figure(figsize=(12, 5))

plt.imshow(epe_visualization, cmap="jet")
plt.colorbar(label="Endpoint error (pixels)")
plt.title("TV-L1 - Endpoint Error Map")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "tvl1_endpoint_error.png"),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. CREATE GROUND TRUTH COMPARISON
# ============================================================

gt_hsv = np.zeros((height, width, 3), dtype=np.uint8)

gt_magnitude_visual = cv2.normalize(
    gt_magnitude, None, 0, 255, cv2.NORM_MINMAX
)

gt_direction = np.degrees(
    np.arctan2(gt_v, gt_u)
) % 360

gt_hsv[:, :, 0] = (gt_direction / 2).astype(np.uint8)
gt_hsv[:, :, 1] = 255
gt_hsv[:, :, 2] = gt_magnitude_visual.astype(np.uint8)

gt_visualization = cv2.cvtColor(
    gt_hsv,
    cv2.COLOR_HSV2BGR
)

comparison = np.hstack((
    cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR),
    gt_visualization,
    flow_visualization
))

cv2.imwrite(
    os.path.join(OUTPUT_DIR, "tvl1_flow_comparison.png"),
    comparison
)


# ============================================================
# 12. CREATE EPE HISTOGRAM
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(valid_epe, bins=60)

plt.xlabel("Endpoint Error (pixels)")
plt.ylabel("Number of pixels")
plt.title("TV-L1 Endpoint Error Distribution")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "tvl1_epe_histogram.png"),
    dpi=200
)

plt.close()


# ============================================================
# 13. SAVE NUMPY ARRAYS
# ============================================================

np.save(
    os.path.join(OUTPUT_DIR, "tvl1_flow.npy"),
    flow
)

np.save(
    os.path.join(OUTPUT_DIR, "tvl1_magnitude.npy"),
    magnitude
)

np.save(
    os.path.join(OUTPUT_DIR, "tvl1_direction.npy"),
    direction
)

np.save(
    os.path.join(OUTPUT_DIR, "tvl1_endpoint_error.npy"),
    epe
)


# ============================================================
# 14. SAVE METRICS CSV
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "tvl1_metrics.csv"),
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(["Metric", "Value"])

    for name, value in metrics.items():
        writer.writerow([name, value])


# ============================================================
# 15. SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "tvl1_summary.txt"
)

with open(summary_path, "w") as file:

    file.write("STAGE 2 - TASK 3\n")
    file.write("TV-L1 DENSE OPTICAL FLOW\n\n")

    file.write("Dataset: MPI Sintel\n")
    file.write("Sequence: alley_1\n")
    file.write("Frames: frame_0001.png -> frame_0002.png\n")
    file.write(f"Image dimensions: {width} x {height}\n\n")

    file.write("TV-L1 parameters:\n")

    for name, value in parameters.items():
        file.write(f"    {name}: {value}\n")

    file.write("\nEvaluation metrics:\n")

    for name, value in metrics.items():
        file.write(f"    {name}: {value}\n")


# ============================================================
# 16. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TV-L1 EVALUATION RESULTS")
print("=" * 60)

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nTask 3 completed successfully.")
print(f"Results saved to: {OUTPUT_DIR}")
