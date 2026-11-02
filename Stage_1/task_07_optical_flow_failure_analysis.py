
"""
STAGE 1 - TASK 7
OPTICAL FLOW LIMITATIONS AND FAILURE ANALYSIS

Dataset: MPI Sintel
Sequence: alley_1
Frame pair: frame_0001 -> frame_0002

Algorithm: Farneback Dense Optical Flow

Analysis:
1. Endpoint error distribution.
2. Error versus ground-truth motion magnitude.
3. Error versus image texture.
4. Spatial distribution of high-error pixels.
5. Error statistics for the raw Sintel occlusion-mask values.

Invalid ground-truth pixels are excluded.
Occlusion-mask values are reported without assuming
a particular binary convention.
"""

from pathlib import Path
import struct
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "MPI-Sintel-training_images"
    / "training"
    / "clean"
    / "alley_1"
    / "frame_0001.png"
)

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

OCCLUSION_MASK_PATH = (
    PROJECT_ROOT
    / "training"
    / "occlusions"
    / "alley_1"
    / "frame_0001.png"
)

FARNEBACK_FLOW_PATH = (
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
    / "Task 7"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. READ SINTEL .FLO FILE
# --------------------------------------------------

def read_flo_file(path):

    with open(path, "rb") as file:

        magic = struct.unpack("<f", file.read(4))[0]

        if magic != 202021.25:
            raise ValueError("Invalid Sintel .flo file.")

        width = struct.unpack("<i", file.read(4))[0]
        height = struct.unpack("<i", file.read(4))[0]

        data = np.fromfile(file, dtype="<f4")

    expected_size = width * height * 2

    if data.size != expected_size:
        raise ValueError("Unexpected ground-truth data size.")

    return data.reshape(height, width, 2)


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

required_files = [
    IMAGE_PATH,
    GROUND_TRUTH_PATH,
    INVALID_MASK_PATH,
    OCCLUSION_MASK_PATH,
    FARNEBACK_FLOW_PATH
]

for path in required_files:

    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

ground_truth = read_flo_file(GROUND_TRUTH_PATH)

estimated_flow = np.load(FARNEBACK_FLOW_PATH)

invalid_mask = cv2.imread(
    str(INVALID_MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

occlusion_mask = cv2.imread(
    str(OCCLUSION_MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if any(item is None for item in [
    image, invalid_mask, occlusion_mask
]):
    raise ValueError("Could not load one or more image masks.")

height, width = image.shape

if ground_truth.shape != (height, width, 2):
    raise ValueError("Ground-truth dimensions do not match image.")

if estimated_flow.shape != ground_truth.shape:
    raise ValueError("Estimated flow dimensions do not match ground truth.")

if invalid_mask.shape != (height, width):
    raise ValueError("Invalid mask dimensions do not match.")

if occlusion_mask.shape != (height, width):
    raise ValueError("Occlusion mask dimensions do not match.")

print("=" * 65)
print("STAGE 1 - TASK 7")
print("OPTICAL FLOW FAILURE ANALYSIS")
print("=" * 65)

print(f"\nImage dimensions: {width} x {height}")


# --------------------------------------------------
# 4. CREATE VALID PIXEL MASK
# --------------------------------------------------

# Sintel invalid-mask convention:
# 0 = valid
# nonzero = invalid

valid_mask = (
    (invalid_mask == 0)
    & np.isfinite(ground_truth).all(axis=2)
    & np.isfinite(estimated_flow).all(axis=2)
)

valid_count = int(np.count_nonzero(valid_mask))

if valid_count == 0:
    raise ValueError("No valid pixels available.")

print(f"Total pixels: {width * height}")
print(f"Valid pixels: {valid_count}")
print(f"Excluded pixels: {width * height - valid_count}")


# --------------------------------------------------
# 5. CALCULATE ENDPOINT ERROR
# --------------------------------------------------

difference = estimated_flow - ground_truth

endpoint_error = np.sqrt(
    difference[..., 0] ** 2
    + difference[..., 1] ** 2
)

valid_epe = endpoint_error[valid_mask]

mean_epe = float(np.mean(valid_epe))
median_epe = float(np.median(valid_epe))
std_epe = float(np.std(valid_epe))

high_error_threshold = float(
    np.percentile(valid_epe, 95)
)

high_error_mask = (
    valid_mask
    & (endpoint_error >= high_error_threshold)
)

print("\nGENERAL ERROR STATISTICS")
print("-" * 45)

print(f"Mean EPE: {mean_epe:.6f} pixels")
print(f"Median EPE: {median_epe:.6f} pixels")
print(f"Standard deviation: {std_epe:.6f} pixels")
print(f"95th percentile EPE: {high_error_threshold:.6f} pixels")
print(f"High-error pixels: {np.count_nonzero(high_error_mask)}")


# --------------------------------------------------
# 6. ANALYSE GROUND-TRUTH MOTION MAGNITUDE
# --------------------------------------------------

gt_u = ground_truth[..., 0]
gt_v = ground_truth[..., 1]

motion_magnitude = np.sqrt(gt_u**2 + gt_v**2)

motion_bins = [0, 1, 3, 5, 10, np.inf]

motion_labels = [
    "0-1 px",
    "1-3 px",
    "3-5 px",
    "5-10 px",
    ">10 px"
]

motion_category = np.digitize(
    motion_magnitude,
    motion_bins[1:-1]
)

motion_statistics = []

for index, label in enumerate(motion_labels):

    region = valid_mask & (motion_category == index)

    if not np.any(region):
        continue

    region_epe = endpoint_error[region]

    motion_statistics.append({
        "Region": label,
        "Pixel Count": int(np.count_nonzero(region)),
        "Mean EPE": float(np.mean(region_epe)),
        "Median EPE": float(np.median(region_epe))
    })

print("\nERROR BY GROUND-TRUTH MOTION MAGNITUDE")
print("-" * 55)

for item in motion_statistics:

    print(
        f"{item['Region']:>10} | "
        f"Pixels: {item['Pixel Count']:>7} | "
        f"Mean EPE: {item['Mean EPE']:.4f}"
    )


# --------------------------------------------------
# 7. ANALYSE IMAGE TEXTURE
# --------------------------------------------------

# Sobel gradients measure local image intensity changes.

image_float = image.astype(np.float32) / 255.0

gradient_x = cv2.Sobel(
    image_float,
    cv2.CV_32F,
    1,
    0,
    ksize=3
)

gradient_y = cv2.Sobel(
    image_float,
    cv2.CV_32F,
    0,
    1,
    ksize=3
)

texture_strength = np.sqrt(
    gradient_x**2 + gradient_y**2
)

valid_texture = texture_strength[valid_mask]

# Divide valid pixels into four texture groups.
texture_quantiles = np.percentile(
    valid_texture,
    [25, 50, 75]
)

texture_category = np.digitize(
    texture_strength,
    texture_quantiles
)

texture_labels = [
    "Low texture",
    "Medium-low texture",
    "Medium-high texture",
    "High texture"
]

texture_statistics = []

for index, label in enumerate(texture_labels):

    region = valid_mask & (texture_category == index)

    if not np.any(region):
        continue

    region_epe = endpoint_error[region]

    texture_statistics.append({
        "Region": label,
        "Pixel Count": int(np.count_nonzero(region)),
        "Mean EPE": float(np.mean(region_epe)),
        "Median EPE": float(np.median(region_epe))
    })

print("\nERROR BY IMAGE TEXTURE")
print("-" * 55)

for item in texture_statistics:

    print(
        f"{item['Region']:>20} | "
        f"Pixels: {item['Pixel Count']:>7} | "
        f"Mean EPE: {item['Mean EPE']:.4f}"
    )


# --------------------------------------------------
# 8. ANALYSE OCCLUSION MASK VALUES
# --------------------------------------------------

# We report the raw mask values rather than assuming
# which value means occluded or non-occluded.

mask_values = np.unique(occlusion_mask)

occlusion_statistics = []

print("\nOCCLUSION MASK ANALYSIS")
print("-" * 55)

print(f"Unique mask values: {mask_values.tolist()}")

for value in mask_values:

    region = (
        valid_mask
        & (occlusion_mask == value)
    )

    if not np.any(region):
        continue

    region_epe = endpoint_error[region]

    item = {
        "Raw mask value": int(value),
        "Pixel Count": int(np.count_nonzero(region)),
        "Mean EPE": float(np.mean(region_epe)),
        "Median EPE": float(np.median(region_epe))
    }

    occlusion_statistics.append(item)

    print(
        f"Mask value {value:>3} | "
        f"Pixels: {item['Pixel Count']:>7} | "
        f"Mean EPE: {item['Mean EPE']:.4f}"
    )


# --------------------------------------------------
# 9. VISUALIZE ENDPOINT ERROR MAP
# --------------------------------------------------

error_visualization = np.zeros(
    (height, width),
    dtype=np.uint8
)

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
    str(RESULTS_DIR / "task7_endpoint_error_map.png"),
    error_color
)


# --------------------------------------------------
# 10. VISUALIZE HIGH-ERROR REGIONS
# --------------------------------------------------

image_color = cv2.cvtColor(
    image,
    cv2.COLOR_GRAY2BGR
)

# Mark high-error pixels in red.
high_error_overlay = image_color.copy()

high_error_overlay[high_error_mask] = (
    0, 0, 255
)

# Blend the original image and error markers.
overlay = cv2.addWeighted(
    image_color,
    0.65,
    high_error_overlay,
    0.35,
    0
)

cv2.imwrite(
    str(RESULTS_DIR / "high_error_regions.png"),
    overlay
)


# --------------------------------------------------
# 11. VISUALIZE MOTION MAGNITUDE
# --------------------------------------------------

motion_visualization = np.zeros(
    (height, width),
    dtype=np.uint8
)

motion_scale = np.percentile(
    motion_magnitude[valid_mask],
    99
)

if motion_scale > 0:

    normalized_motion = np.clip(
        motion_magnitude / motion_scale,
        0,
        1
    )

    motion_visualization[valid_mask] = (
        normalized_motion[valid_mask] * 255
    ).astype(np.uint8)

motion_color = cv2.applyColorMap(
    motion_visualization,
    cv2.COLORMAP_JET
)

motion_color[~valid_mask] = 0

cv2.imwrite(
    str(RESULTS_DIR / "ground_truth_motion_magnitude.png"),
    motion_color
)


# --------------------------------------------------
# 12. PLOT ERROR BY MOTION MAGNITUDE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    [item["Region"] for item in motion_statistics],
    [item["Mean EPE"] for item in motion_statistics]
)

plt.xlabel("Ground-Truth Motion Magnitude")
plt.ylabel("Mean Endpoint Error (pixels)")
plt.title("Optical Flow Error by Motion Magnitude")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "error_by_motion_magnitude.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 13. PLOT ERROR BY IMAGE TEXTURE
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    [item["Region"] for item in texture_statistics],
    [item["Mean EPE"] for item in texture_statistics]
)

plt.xlabel("Image Texture Group")
plt.ylabel("Mean Endpoint Error (pixels)")
plt.title("Optical Flow Error by Image Texture")
plt.xticks(rotation=15)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "error_by_texture.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 14. SAVE ALL ANALYSIS STATISTICS
# --------------------------------------------------

csv_path = RESULTS_DIR / "failure_analysis_metrics.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Analysis",
        "Region",
        "Pixel Count",
        "Mean EPE",
        "Median EPE"
    ])

    for item in motion_statistics:

        writer.writerow([
            "Motion magnitude",
            item["Region"],
            item["Pixel Count"],
            item["Mean EPE"],
            item["Median EPE"]
        ])

    for item in texture_statistics:

        writer.writerow([
            "Image texture",
            item["Region"],
            item["Pixel Count"],
            item["Mean EPE"],
            item["Median EPE"]
        ])

    for item in occlusion_statistics:

        writer.writerow([
            "Raw occlusion mask value",
            item["Raw mask value"],
            item["Pixel Count"],
            item["Mean EPE"],
            item["Median EPE"]
        ])


# --------------------------------------------------
# 15. SAVE TEXT SUMMARY
# --------------------------------------------------

summary_lines = [
    "STAGE 1 - TASK 7",
    "OPTICAL FLOW LIMITATIONS AND FAILURE ANALYSIS",
    "",
    "Dataset: MPI Sintel",
    "Sequence: alley_1",
    "Frame pair: frame_0001 -> frame_0002",
    "Algorithm: Farneback Dense Optical Flow",
    "",
    f"Image dimensions: {width} x {height}",
    f"Total pixels: {width * height}",
    f"Valid pixels: {valid_count}",
    "",
    "GENERAL ERROR STATISTICS",
    f"Mean EPE: {mean_epe:.6f}",
    f"Median EPE: {median_epe:.6f}",
    f"Standard deviation: {std_epe:.6f}",
    f"95th percentile EPE: {high_error_threshold:.6f}",
    f"High-error pixel count: {np.count_nonzero(high_error_mask)}",
    "",
    "ERROR BY GROUND-TRUTH MOTION MAGNITUDE"
]

for item in motion_statistics:

    summary_lines.append(
        f"{item['Region']}: "
        f"Pixels={item['Pixel Count']}, "
        f"Mean EPE={item['Mean EPE']:.6f}, "
        f"Median EPE={item['Median EPE']:.6f}"
    )

summary_lines.extend([
    "",
    "ERROR BY IMAGE TEXTURE"
])

for item in texture_statistics:

    summary_lines.append(
        f"{item['Region']}: "
        f"Pixels={item['Pixel Count']}, "
        f"Mean EPE={item['Mean EPE']:.6f}, "
        f"Median EPE={item['Median EPE']:.6f}"
    )

summary_lines.extend([
    "",
    "OCCLUSION MASK ANALYSIS",
    f"Raw mask values: {mask_values.tolist()}",
    "The raw mask values are reported without assuming",
    "their semantic convention."
])

for item in occlusion_statistics:

    summary_lines.append(
        f"Mask value {item['Raw mask value']}: "
        f"Pixels={item['Pixel Count']}, "
        f"Mean EPE={item['Mean EPE']:.6f}, "
        f"Median EPE={item['Median EPE']:.6f}"
    )

summary_lines.extend([
    "",
    "NOTES",
    "Invalid ground-truth pixels were excluded.",
    "Texture groups are defined by quartiles of Sobel gradient magnitude.",
    "Motion groups are defined by ground-truth motion magnitude.",
    "High-error regions are defined by the 95th percentile of valid EPE.",
    "This experiment analyses one Sintel frame pair only."
])

(RESULTS_DIR / "failure_analysis_summary.txt").write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


# --------------------------------------------------
# 16. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TASK 7 COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"\nResults saved in:\n{RESULTS_DIR}")

print("\nGenerated files:")
print("  - task7_endpoint_error_map.png")
print("  - high_error_regions.png")
print("  - ground_truth_motion_magnitude.png")
print("  - error_by_motion_magnitude.png")
print("  - error_by_texture.png")
print("  - failure_analysis_metrics.csv")
print("  - failure_analysis_summary.txt")
