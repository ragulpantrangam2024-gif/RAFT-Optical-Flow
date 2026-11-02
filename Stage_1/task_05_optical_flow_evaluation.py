
"""
STAGE 1 - TASK 5
OPTICAL FLOW EVALUATION USING GROUND TRUTH

Dataset: MPI Sintel
Sequence: alley_1
Frame pair: frame_0001.png -> frame_0002.png

Ground truth:
    training/flow/alley_1/frame_0001.flo

Estimated flow:
    Stage_1/results/Task 4/farneback_flow.npy

Evaluation:
    - Endpoint Error (EPE)
    - Average Endpoint Error (AEPE)
    - Median EPE
    - Percentage of pixels below EPE thresholds
    - Average Angular Error (AAE)

Invalid ground-truth pixels are excluded using Sintel's
invalid-pixel mask.
"""

from pathlib import Path
import struct

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "training"
    / "flow"
    / "alley_1"
    / "frame_0001.flo"
)

INVALID_MASK_PATH = (
    PROJECT_ROOT
    / "training"
    / "invalid"
    / "alley_1"
    / "frame_0001.png"
)

ESTIMATED_FLOW_PATH = (
    PROJECT_ROOT
    / "Stage_1"
    / "results"
    / "Task 4"
    / "farneback_flow.npy"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "Stage_1"
    / "results"
    / "Task 5"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. READ SINTEL .FLO FILE
# --------------------------------------------------

def read_flo_file(path):
    """
    Read a Middlebury .flo optical flow file.

    File structure:
        4 bytes: magic number
        4 bytes: width
        4 bytes: height
        Remaining: float32 horizontal and vertical flow
    """

    with open(path, "rb") as file:

        magic = struct.unpack("<f", file.read(4))[0]

        if magic != 202021.25:
            raise ValueError("Invalid .flo file: incorrect magic number.")

        width = struct.unpack("<i", file.read(4))[0]
        height = struct.unpack("<i", file.read(4))[0]

        data = np.fromfile(file, dtype="<f4")

    expected_values = height * width * 2

    if data.size != expected_values:
        raise ValueError(
            f"Incorrect .flo data size. Expected {expected_values}, "
            f"received {data.size}."
        )

    flow = data.reshape(height, width, 2)

    return flow


# --------------------------------------------------
# 3. LOAD GROUND TRUTH AND ESTIMATED FLOW
# --------------------------------------------------

if not GROUND_TRUTH_PATH.exists():
    raise FileNotFoundError(GROUND_TRUTH_PATH)

if not ESTIMATED_FLOW_PATH.exists():
    raise FileNotFoundError(ESTIMATED_FLOW_PATH)

ground_truth_flow = read_flo_file(GROUND_TRUTH_PATH)

estimated_flow = np.load(ESTIMATED_FLOW_PATH)

if ground_truth_flow.shape != estimated_flow.shape:
    raise ValueError(
        f"Flow shape mismatch: Ground truth "
        f"{ground_truth_flow.shape}, Estimated {estimated_flow.shape}"
    )

height, width, _ = ground_truth_flow.shape

print("=" * 65)
print("STAGE 1 - TASK 5")
print("OPTICAL FLOW EVALUATION")
print("=" * 65)

print(f"\nImage dimensions: {width} x {height}")
print(f"Ground truth shape: {ground_truth_flow.shape}")
print(f"Estimated flow shape: {estimated_flow.shape}")


# --------------------------------------------------
# 4. CREATE VALID PIXEL MASK
# --------------------------------------------------

# Sintel invalid mask convention:
# 0   = valid pixel
# 255 = invalid pixel

if INVALID_MASK_PATH.exists():

    invalid_mask = cv2.imread(
        str(INVALID_MASK_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    if invalid_mask is None:
        raise ValueError("Could not read the invalid-pixel mask.")

    if invalid_mask.shape != (height, width):
        raise ValueError("Invalid mask dimensions do not match flow.")

    valid_mask = invalid_mask == 0

    print("\nInvalid-pixel mask loaded successfully.")

else:

    print("\nWARNING: Invalid mask not found.")
    print("Using finite ground-truth flow values only.")

    valid_mask = np.ones((height, width), dtype=bool)


# Exclude non-finite values from both flow fields.
finite_mask = (
    np.isfinite(ground_truth_flow).all(axis=2)
    & np.isfinite(estimated_flow).all(axis=2)
)

valid_mask = valid_mask & finite_mask

if not np.any(valid_mask):
    raise ValueError("No valid pixels available for evaluation.")

valid_count = int(np.count_nonzero(valid_mask))
total_count = height * width

print(f"Total pixels: {total_count}")
print(f"Valid pixels: {valid_count}")
print(f"Excluded pixels: {total_count - valid_count}")
print(f"Valid pixel percentage: {100 * valid_count / total_count:.2f}%")


# --------------------------------------------------
# 5. CALCULATE ENDPOINT ERROR
# --------------------------------------------------

difference = estimated_flow - ground_truth_flow

endpoint_error = np.sqrt(
    difference[..., 0] ** 2
    + difference[..., 1] ** 2
)

valid_epe = endpoint_error[valid_mask]

mean_epe = float(np.mean(valid_epe))
median_epe = float(np.median(valid_epe))
max_epe = float(np.max(valid_epe))
min_epe = float(np.min(valid_epe))
std_epe = float(np.std(valid_epe))

# Percentage of valid pixels below each threshold.
below_1 = 100 * np.mean(valid_epe < 1.0)
below_3 = 100 * np.mean(valid_epe < 3.0)
below_5 = 100 * np.mean(valid_epe < 5.0)


# --------------------------------------------------
# 6. CALCULATE AVERAGE ANGULAR ERROR
# --------------------------------------------------

gt_u = ground_truth_flow[..., 0]
gt_v = ground_truth_flow[..., 1]

est_u = estimated_flow[..., 0]
est_v = estimated_flow[..., 1]

gt_magnitude = np.sqrt(gt_u**2 + gt_v**2)
est_magnitude = np.sqrt(est_u**2 + est_v**2)

# Angular error is unstable when either vector is near zero.
direction_mask = (
    valid_mask
    & (gt_magnitude > 1e-6)
    & (est_magnitude > 1e-6)
)

if np.any(direction_mask):

    dot_product = (
        gt_u[direction_mask] * est_u[direction_mask]
        + gt_v[direction_mask] * est_v[direction_mask]
    )

    denominator = (
        gt_magnitude[direction_mask]
        * est_magnitude[direction_mask]
    )

    cosine_similarity = dot_product / denominator

    cosine_similarity = np.clip(
        cosine_similarity,
        -1.0,
        1.0
    )

    angular_errors = np.degrees(
        np.arccos(cosine_similarity)
    )

    mean_angular_error = float(np.mean(angular_errors))

else:

    mean_angular_error = float("nan")


# --------------------------------------------------
# 7. PRINT EVALUATION RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("ENDPOINT ERROR (EPE)")
print("=" * 65)

print(f"Mean EPE:   {mean_epe:.6f} pixels")
print(f"Median EPE: {median_epe:.6f} pixels")
print(f"Maximum EPE:{max_epe:.6f} pixels")
print(f"Minimum EPE:{min_epe:.6f} pixels")
print(f"Std Dev:    {std_epe:.6f} pixels")

print("\nPIXEL ACCURACY")
print("-" * 45)

print(f"EPE < 1 pixel: {below_1:.2f}%")
print(f"EPE < 3 pixels: {below_3:.2f}%")
print(f"EPE < 5 pixels: {below_5:.2f}%")

print("\nANGULAR ERROR")
print("-" * 45)

print(f"Mean Angular Error: {mean_angular_error:.6f} degrees")


# --------------------------------------------------
# 8. CREATE ENDPOINT ERROR MAP
# --------------------------------------------------

# Invalid pixels are displayed in black.
error_visualization = np.zeros(
    (height, width),
    dtype=np.uint8
)

# Use the 99th percentile to make the visualization
# readable without letting extreme errors dominate.
error_scale = np.percentile(valid_epe, 99)

if error_scale > 0:

    normalized_error = np.clip(
        endpoint_error / error_scale,
        0,
        1
    )

    error_visualization[valid_mask] = (
        normalized_error[valid_mask] * 255
    ).astype(np.uint8)

error_color = cv2.applyColorMap(
    error_visualization,
    cv2.COLORMAP_JET
)

error_color[~valid_mask] = 0

cv2.imwrite(
    str(RESULTS_DIR / "endpoint_error_map.png"),
    error_color
)


# --------------------------------------------------
# 9. CREATE EPE HISTOGRAM
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    valid_epe,
    bins=100,
    range=(0, min(float(np.percentile(valid_epe, 99)), 20)),
    edgecolor="black"
)

plt.xlabel("Endpoint Error (pixels)")
plt.ylabel("Number of Pixels")
plt.title("Farneback Optical Flow - Endpoint Error Distribution")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "endpoint_error_histogram.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 10. CREATE GROUND TRUTH VS ESTIMATED FLOW
# --------------------------------------------------

def flow_to_color(flow, magnitude_scale):
    """
    Convert flow to HSV visualization.

    Hue: direction
    Saturation: maximum
    Value: normalized magnitude
    """

    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        u,
        v,
        angleInDegrees=True
    )

    hsv = np.zeros(
        (flow.shape[0], flow.shape[1], 3),
        dtype=np.uint8
    )

    hsv[..., 0] = np.mod(angle / 2, 180).astype(np.uint8)
    hsv[..., 1] = 255

    if magnitude_scale > 0:

        normalized = np.clip(
            magnitude / magnitude_scale,
            0,
            1
        )

    else:
        normalized = np.zeros_like(magnitude)

    hsv[..., 2] = (normalized * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Use one shared magnitude scale for a fairer visual comparison.
combined_magnitude = np.concatenate((
    gt_magnitude[valid_mask],
    est_magnitude[valid_mask]
))

shared_scale = float(np.percentile(combined_magnitude, 99))

gt_color = flow_to_color(ground_truth_flow, shared_scale)
estimated_color = flow_to_color(estimated_flow, shared_scale)

comparison = np.hstack((gt_color, estimated_color))

cv2.imwrite(
    str(RESULTS_DIR / "flow_comparison.png"),
    comparison
)


# --------------------------------------------------
# 11. SAVE EVALUATION METRICS
# --------------------------------------------------

metrics = {
    "metric": [
        "Valid pixels",
        "Excluded pixels",
        "Mean EPE (pixels)",
        "Median EPE (pixels)",
        "Maximum EPE (pixels)",
        "Minimum EPE (pixels)",
        "EPE standard deviation",
        "Pixels with EPE < 1 px (%)",
        "Pixels with EPE < 3 px (%)",
        "Pixels with EPE < 5 px (%)",
        "Mean Angular Error (degrees)"
    ],
    "value": [
        valid_count,
        total_count - valid_count,
        mean_epe,
        median_epe,
        max_epe,
        min_epe,
        std_epe,
        below_1,
        below_3,
        below_5,
        mean_angular_error
    ]
}

import csv

with open(
    RESULTS_DIR / "evaluation_metrics.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])

    for metric, value in zip(
        metrics["metric"],
        metrics["value"]
    ):
        writer.writerow([metric, value])


# --------------------------------------------------
# 12. SAVE EVALUATION SUMMARY
# --------------------------------------------------

summary = f"""
STAGE 1 - TASK 5
OPTICAL FLOW EVALUATION

Dataset: MPI Sintel
Sequence: alley_1
Frame pair: frame_0001 -> frame_0002

Estimated algorithm: Farneback Dense Optical Flow

Ground truth:
    {GROUND_TRUTH_PATH}

Estimated flow:
    {ESTIMATED_FLOW_PATH}

Image dimensions:
    {width} x {height}

Pixel statistics:
    Total pixels: {total_count}
    Valid pixels: {valid_count}
    Excluded pixels: {total_count - valid_count}
    Valid percentage: {100 * valid_count / total_count:.4f}%

Endpoint Error:
    Mean EPE: {mean_epe:.6f} pixels
    Median EPE: {median_epe:.6f} pixels
    Maximum EPE: {max_epe:.6f} pixels
    Minimum EPE: {min_epe:.6f} pixels
    Standard deviation: {std_epe:.6f} pixels

Pixel accuracy:
    EPE < 1 pixel: {below_1:.4f}%
    EPE < 3 pixels: {below_3:.4f}%
    EPE < 5 pixels: {below_5:.4f}%

Angular error:
    Mean Angular Error: {mean_angular_error:.6f} degrees

Evaluation:
    Invalid ground-truth pixels were excluded.
    Angular error excludes near-zero flow vectors.
"""

(RESULTS_DIR / "evaluation_summary.txt").write_text(
    summary,
    encoding="utf-8"
)


# --------------------------------------------------
# 13. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TASK 5 COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"\nResults saved in:\n{RESULTS_DIR}")

print("\nGenerated files:")
print("  - endpoint_error_map.png")
print("  - endpoint_error_histogram.png")
print("  - flow_comparison.png")
print("  - evaluation_metrics.csv")
print("  - evaluation_summary.txt")
