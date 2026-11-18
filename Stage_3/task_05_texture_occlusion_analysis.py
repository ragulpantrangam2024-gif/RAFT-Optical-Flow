
"""
STAGE 3 - TASK 5
TEXTURE AND OCCLUSION-BASED OPTICAL FLOW ERROR ANALYSIS

Dataset: KITTI 2015 Optical Flow
Sample: 000000_10.png -> 000000_11.png

Experiments:
1. Evaluate optical flow in low, medium and high texture regions.
2. Evaluate optical flow in occluded and non-occluded regions.
3. Compare Farneback, Horn-Schunck, TV-L1 and DIS Medium.

Outputs:
Stage_3/results/Task 5/
"""

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

KITTI_ROOT = (
    PROJECT_ROOT
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
)

IMAGE_1_PATH = KITTI_ROOT / "image_2" / "000000_10.png"
IMAGE_2_PATH = KITTI_ROOT / "image_2" / "000000_11.png"

FLOW_OCC_PATH = KITTI_ROOT / "flow_occ" / "000000_10.png"
FLOW_NOC_PATH = KITTI_ROOT / "flow_noc" / "000000_10.png"

OUTPUT_DIR = PROJECT_ROOT / "Stage_3" / "results" / "Task 5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HS_ALPHA = 1.0
HS_ITERATIONS = 100

TEXTURE_QUANTILES = (1 / 3, 2 / 3)


# ============================================================
# 2. LOAD KITTI IMAGES AND GROUND TRUTH
# ============================================================

def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return image


def load_kitti_flow(path):
    """
    KITTI flow PNG encoding:
    - Channel 0: horizontal flow, encoded with offset 32768 and scale 64
    - Channel 1: vertical flow, encoded with offset 32768 and scale 64
    - Channel 2: validity flag
    """

    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if encoded is None:
        raise FileNotFoundError(f"Could not load flow file: {path}")

    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"Unexpected KITTI flow format: {path}")

    encoded = encoded.astype(np.float32)

    flow = np.zeros(
        (encoded.shape[0], encoded.shape[1], 2),
        dtype=np.float32
    )

    flow[..., 0] = (encoded[..., 2] - 32768.0) / 64.0
    flow[..., 1] = (encoded[..., 1] - 32768.0) / 64.0

    valid = encoded[..., 0] > 0

    return flow, valid


print("=" * 70)
print("STAGE 3 - TASK 5: TEXTURE AND OCCLUSION ANALYSIS")
print("=" * 70)

print("\nLoading KITTI images...")

image_1 = load_image(IMAGE_1_PATH)
image_2 = load_image(IMAGE_2_PATH)

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

print("Image shape:", gray_1.shape)

print("\nLoading KITTI ground truth...")

gt_flow, valid_occ = load_kitti_flow(FLOW_OCC_PATH)
gt_flow_noc, valid_noc = load_kitti_flow(FLOW_NOC_PATH)

if gray_1.shape != gray_2.shape:
    raise ValueError("Input image dimensions do not match.")

if gt_flow.shape[:2] != gray_1.shape:
    raise ValueError("Ground-truth and image dimensions do not match.")

valid_pixels = valid_occ

occluded_mask = valid_occ & (~valid_noc)
non_occluded_mask = valid_occ & valid_noc

print("Valid pixels:", int(np.count_nonzero(valid_pixels)))
print("Occluded pixels:", int(np.count_nonzero(occluded_mask)))
print("Non-occluded pixels:", int(np.count_nonzero(non_occluded_mask)))


# ============================================================
# 3. TEXTURE ANALYSIS USING SOBEL GRADIENTS
# ============================================================

def calculate_texture(gray_image):
    """
    Calculate image gradient magnitude using Sobel derivatives.
    Higher gradient magnitude indicates stronger local intensity changes.
    """

    gray = gray_image.astype(np.float32) / 255.0

    gradient_x = cv2.Sobel(
        gray, cv2.CV_32F, 1, 0, ksize=3
    )

    gradient_y = cv2.Sobel(
        gray, cv2.CV_32F, 0, 1, ksize=3
    )

    magnitude = cv2.magnitude(gradient_x, gradient_y)

    return magnitude


texture_map = calculate_texture(gray_1)

valid_texture_values = texture_map[valid_pixels]

if valid_texture_values.size == 0:
    raise ValueError("No valid KITTI ground-truth pixels found.")

low_threshold, high_threshold = np.quantile(
    valid_texture_values,
    TEXTURE_QUANTILES
)

low_texture_mask = (
    valid_pixels & (texture_map <= low_threshold)
)

medium_texture_mask = (
    valid_pixels
    & (texture_map > low_threshold)
    & (texture_map <= high_threshold)
)

high_texture_mask = (
    valid_pixels & (texture_map > high_threshold)
)

texture_masks = {
    "Low Texture": low_texture_mask,
    "Medium Texture": medium_texture_mask,
    "High Texture": high_texture_mask
}

print("\nTexture thresholds:")
print(f"Low/Medium threshold: {low_threshold:.6f}")
print(f"Medium/High threshold: {high_threshold:.6f}")

for name, mask in texture_masks.items():
    print(f"{name}: {np.count_nonzero(mask)} pixels")


# ============================================================
# 4. OPTICAL FLOW ALGORITHMS
# ============================================================

def calculate_farneback(first, second):
    return cv2.calcOpticalFlowFarneback(
        first,
        second,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )


def calculate_horn_schunck(first, second):
    """
    Basic single-scale Horn-Schunck implementation.
    """

    first = first.astype(np.float32) / 255.0
    second = second.astype(np.float32) / 255.0

    ix = cv2.Sobel(first, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    iy = cv2.Sobel(first, cv2.CV_32F, 0, 1, ksize=3) / 8.0

    it = second - first

    u = np.zeros_like(first, dtype=np.float32)
    v = np.zeros_like(first, dtype=np.float32)

    kernel = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6,  0,    1 / 6],
            [1 / 12, 1 / 6, 1 / 12]
        ],
        dtype=np.float32
    )

    alpha_squared = HS_ALPHA ** 2

    for _ in range(HS_ITERATIONS):

        u_avg = cv2.filter2D(
            u, -1, kernel, borderType=cv2.BORDER_REFLECT
        )

        v_avg = cv2.filter2D(
            v, -1, kernel, borderType=cv2.BORDER_REFLECT
        )

        correction = (
            ix * u_avg + iy * v_avg + it
        ) / (
            alpha_squared + ix ** 2 + iy ** 2
        )

        u = u_avg - ix * correction
        v = v_avg - iy * correction

    return np.dstack((u, v)).astype(np.float32)


def calculate_tvl1(first, second):
    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            "TV-L1 requires opencv-contrib-python."
        )

    if not hasattr(cv2.optflow, "createOptFlow_DualTVL1"):
        raise RuntimeError(
            "OpenCV TV-L1 implementation is unavailable."
        )

    algorithm = cv2.optflow.createOptFlow_DualTVL1()

    algorithm.setTau(0.25)
    algorithm.setLambda(0.15)
    algorithm.setTheta(0.3)
    algorithm.setScalesNumber(5)
    algorithm.setWarpingsNumber(5)
    algorithm.setEpsilon(0.01)
    algorithm.setInnerIterations(30)
    algorithm.setOuterIterations(10)
    algorithm.setScaleStep(0.8)
    algorithm.setGamma(0.0)
    algorithm.setMedianFiltering(5)

    return algorithm.calc(first, second, None)


def calculate_dis(first, second):
    algorithm = cv2.DISOpticalFlow_create(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    )

    algorithm.setUseSpatialPropagation(True)
    algorithm.setVariationalRefinementIterations(5)
    algorithm.setVariationalRefinementAlpha(20.0)
    algorithm.setVariationalRefinementDelta(5.0)
    algorithm.setVariationalRefinementGamma(10.0)

    return algorithm.calc(first, second, None)


# ============================================================
# 5. CALCULATE OPTICAL FLOW AND RUNTIME
# ============================================================

algorithms = {
    "Farneback": calculate_farneback,
    "Horn-Schunck": calculate_horn_schunck,
    "TV-L1": calculate_tvl1,
    "DIS Medium": calculate_dis
}

predicted_flows = {}
runtime_results = {}

print("\nCalculating optical flow...")

for name, function in algorithms.items():

    print(f"\nRunning {name}...")

    start_time = time.perf_counter()

    try:
        predicted_flow = function(gray_1, gray_2)
    except Exception as error:
        raise RuntimeError(
            f"{name} failed. Check its OpenCV implementation. "
            f"Original error: {error}"
        ) from error

    elapsed = time.perf_counter() - start_time

    if predicted_flow.shape != gt_flow.shape:
        raise ValueError(
            f"{name} returned an unexpected flow shape: "
            f"{predicted_flow.shape}"
        )

    predicted_flows[name] = predicted_flow
    runtime_results[name] = elapsed

    print(f"Runtime: {elapsed:.6f} seconds")


# ============================================================
# 6. CALCULATE ENDPOINT ERROR AND KITTI FL-ALL
# ============================================================

def calculate_error(predicted, ground_truth):
    difference = predicted - ground_truth

    endpoint_error = np.sqrt(
        difference[..., 0] ** 2
        + difference[..., 1] ** 2
    )

    return endpoint_error


def calculate_metrics(error_map, ground_truth, mask):
    """
    KITTI Fl-all:
    EPE > 3 pixels AND relative EPE > 5% of GT flow magnitude.
    """

    count = int(np.count_nonzero(mask))

    if count == 0:
        return {
            "Valid Pixels": 0,
            "Mean EPE": np.nan,
            "Median EPE": np.nan,
            "Fl-all (%)": np.nan,
            "EPE Below 3px (%)": np.nan
        }

    errors = error_map[mask]

    gt_magnitude = np.linalg.norm(
        ground_truth, axis=2
    )

    relative_error = errors / np.maximum(
        gt_magnitude[mask], 1e-6
    )

    outliers = (
        (errors > 3.0)
        & (relative_error > 0.05)
    )

    return {
        "Valid Pixels": count,
        "Mean EPE": float(np.mean(errors)),
        "Median EPE": float(np.median(errors)),
        "Fl-all (%)": float(np.mean(outliers) * 100.0),
        "EPE Below 3px (%)": float(
            np.mean(errors <= 3.0) * 100.0
        )
    }


# ============================================================
# 7. TEXTURE-BASED ERROR EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEXTURE-BASED ERROR ANALYSIS")
print("=" * 70)

results = []

error_maps = {}

for algorithm_name, predicted_flow in predicted_flows.items():

    error_map = calculate_error(predicted_flow, gt_flow)
    error_maps[algorithm_name] = error_map

    for category, mask in texture_masks.items():

        metrics = calculate_metrics(
            error_map,
            gt_flow,
            mask
        )

        row = {
            "Algorithm": algorithm_name,
            "Analysis": "Texture",
            "Category": category,
            **metrics
        }

        results.append(row)

        print(f"\n{algorithm_name} - {category}")
        print(f"Valid pixels: {metrics['Valid Pixels']}")
        print(f"Mean EPE: {metrics['Mean EPE']:.6f}")
        print(f"Fl-all: {metrics['Fl-all (%)']:.6f}%")


# ============================================================
# 8. OCCLUSION-BASED ERROR EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("OCCLUSION-BASED ERROR ANALYSIS")
print("=" * 70)

occlusion_masks = {
    "Non-Occluded": non_occluded_mask,
    "Occluded": occluded_mask
}

for algorithm_name, predicted_flow in predicted_flows.items():

    error_map = error_maps[algorithm_name]

    for category, mask in occlusion_masks.items():

        metrics = calculate_metrics(
            error_map,
            gt_flow,
            mask
        )

        row = {
            "Algorithm": algorithm_name,
            "Analysis": "Occlusion",
            "Category": category,
            **metrics
        }

        results.append(row)

        print(f"\n{algorithm_name} - {category}")
        print(f"Valid pixels: {metrics['Valid Pixels']}")
        print(f"Mean EPE: {metrics['Mean EPE']:.6f}")
        print(f"Fl-all: {metrics['Fl-all (%)']:.6f}%")


# ============================================================
# 9. SAVE CSV RESULTS
# ============================================================

df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "texture_occlusion_metrics.csv"
df.to_csv(csv_path, index=False)

print("\nCSV saved:", csv_path)


# ============================================================
# 10. VISUALIZATION - MEAN EPE BY TEXTURE
# ============================================================

algorithm_names = list(algorithms.keys())
texture_categories = list(texture_masks.keys())

x = np.arange(len(texture_categories))
width = 0.19

fig, ax = plt.subplots(figsize=(12, 6))

for index, algorithm_name in enumerate(algorithm_names):

    values = []

    for category in texture_categories:

        selected = df[
            (df["Algorithm"] == algorithm_name)
            & (df["Analysis"] == "Texture")
            & (df["Category"] == category)
        ]

        values.append(selected["Mean EPE"].iloc[0])

    offset = (index - 1.5) * width

    ax.bar(
        x + offset,
        values,
        width,
        label=algorithm_name
    )

ax.set_title("Mean EPE Across Texture Categories")
ax.set_xlabel("Image Texture")
ax.set_ylabel("Mean Endpoint Error (pixels)")
ax.set_xticks(x)
ax.set_xticklabels(texture_categories)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

texture_plot = OUTPUT_DIR / "mean_epe_by_texture.png"
plt.savefig(texture_plot, dpi=200)
plt.close()

print("Saved:", texture_plot)


# ============================================================
# 11. VISUALIZATION - FL-ALL BY TEXTURE
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

for index, algorithm_name in enumerate(algorithm_names):

    values = []

    for category in texture_categories:

        selected = df[
            (df["Algorithm"] == algorithm_name)
            & (df["Analysis"] == "Texture")
            & (df["Category"] == category)
        ]

        values.append(selected["Fl-all (%)"].iloc[0])

    offset = (index - 1.5) * width

    ax.bar(
        x + offset,
        values,
        width,
        label=algorithm_name
    )

ax.set_title("KITTI Fl-all Across Texture Categories")
ax.set_xlabel("Image Texture")
ax.set_ylabel("Fl-all Outliers (%)")
ax.set_xticks(x)
ax.set_xticklabels(texture_categories)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

texture_fl_plot = OUTPUT_DIR / "fl_all_by_texture.png"
plt.savefig(texture_fl_plot, dpi=200)
plt.close()

print("Saved:", texture_fl_plot)


# ============================================================
# 12. VISUALIZATION - OCCLUSION COMPARISON
# ============================================================

occlusion_categories = list(occlusion_masks.keys())
x_occ = np.arange(len(occlusion_categories))
width_occ = 0.18

fig, ax = plt.subplots(figsize=(10, 6))

for index, algorithm_name in enumerate(algorithm_names):

    values = []

    for category in occlusion_categories:

        selected = df[
            (df["Algorithm"] == algorithm_name)
            & (df["Analysis"] == "Occlusion")
            & (df["Category"] == category)
        ]

        values.append(selected["Mean EPE"].iloc[0])

    offset = (index - 1.5) * width_occ

    ax.bar(
        x_occ + offset,
        values,
        width_occ,
        label=algorithm_name
    )

ax.set_title("Mean EPE: Occluded vs Non-Occluded Regions")
ax.set_xlabel("Region Type")
ax.set_ylabel("Mean Endpoint Error (pixels)")
ax.set_xticks(x_occ)
ax.set_xticklabels(occlusion_categories)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

occlusion_plot = OUTPUT_DIR / "mean_epe_occlusion_comparison.png"
plt.savefig(occlusion_plot, dpi=200)
plt.close()

print("Saved:", occlusion_plot)


# ============================================================
# 13. VISUALIZATION - OCCLUSION MASK
# ============================================================

occlusion_visual = np.zeros(
    gray_1.shape,
    dtype=np.uint8
)

occlusion_visual[non_occluded_mask] = 1
occlusion_visual[occluded_mask] = 2

fig, ax = plt.subplots(figsize=(14, 5))

im = ax.imshow(
    occlusion_visual,
    cmap="viridis",
    interpolation="nearest",
    vmin=0,
    vmax=2
)

ax.set_title("KITTI Occlusion Classification")
ax.set_xlabel("Image X")
ax.set_ylabel("Image Y")

colorbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
colorbar.ax.set_yticklabels([
    "Invalid / Unlabeled",
    "Non-Occluded",
    "Occluded"
])

plt.tight_layout()

occlusion_mask_path = OUTPUT_DIR / "kitti_occlusion_mask.png"
plt.savefig(occlusion_mask_path, dpi=200)
plt.close()

print("Saved:", occlusion_mask_path)


# ============================================================
# 14. VISUALIZATION - TEXTURE MAP
# ============================================================

texture_visual = np.zeros(
    gray_1.shape,
    dtype=np.uint8
)

texture_visual[low_texture_mask] = 1
texture_visual[medium_texture_mask] = 2
texture_visual[high_texture_mask] = 3

fig, ax = plt.subplots(figsize=(14, 5))

im = ax.imshow(
    texture_visual,
    cmap="viridis",
    interpolation="nearest",
    vmin=0,
    vmax=3
)

ax.set_title("KITTI Image Texture Classification")
ax.set_xlabel("Image X")
ax.set_ylabel("Image Y")

colorbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
colorbar.ax.set_yticklabels([
    "Invalid",
    "Low",
    "Medium",
    "High"
])

plt.tight_layout()

texture_map_path = OUTPUT_DIR / "kitti_texture_map.png"
plt.savefig(texture_map_path, dpi=200)
plt.close()

print("Saved:", texture_map_path)


# ============================================================
# 15. SAVE SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "texture_occlusion_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 3 - TASK 5: TEXTURE AND OCCLUSION ANALYSIS\n")
    file.write("=" * 65 + "\n\n")

    file.write("Dataset: KITTI 2015 training\n")
    file.write("Sample: 000000_10.png -> 000000_11.png\n")
    file.write(f"Image dimensions: {gray_1.shape[1]} x {gray_1.shape[0]}\n")
    file.write(f"Valid pixels: {np.count_nonzero(valid_pixels)}\n")
    file.write(f"Occluded pixels: {np.count_nonzero(occluded_mask)}\n")
    file.write(f"Non-occluded pixels: {np.count_nonzero(non_occluded_mask)}\n\n")

    file.write("Texture thresholds:\n")
    file.write(f"Low/Medium: {low_threshold:.8f}\n")
    file.write(f"Medium/High: {high_threshold:.8f}\n\n")

    file.write("Texture pixel counts:\n")

    for category, mask in texture_masks.items():
        file.write(
            f"{category}: {np.count_nonzero(mask)}\n"
        )

    file.write("\nAlgorithm runtimes:\n")

    for name, runtime in runtime_results.items():
        file.write(f"{name}: {runtime:.6f} seconds\n")

    file.write("\nDetailed evaluation results:\n")
    file.write("=" * 65 + "\n")

    for _, row in df.iterrows():

        file.write(
            f"\n{row['Algorithm']} - "
            f"{row['Analysis']} - {row['Category']}\n"
        )

        file.write(f"Valid pixels: {row['Valid Pixels']}\n")
        file.write(f"Mean EPE: {row['Mean EPE']:.6f}\n")
        file.write(f"Median EPE: {row['Median EPE']:.6f}\n")
        file.write(f"Fl-all: {row['Fl-all (%)']:.6f}%\n")
        file.write(
            f"EPE below 3px: "
            f"{row['EPE Below 3px (%)']:.6f}%\n"
        )

    file.write("\nImportant notes:\n")
    file.write(
        "- Texture is classified using Sobel gradient magnitude "
        "from the first image.\n"
    )
    file.write(
        "- Texture categories use tertiles of valid-pixel gradient "
        "magnitudes.\n"
    )
    file.write(
        "- Occluded pixels are valid in flow_occ but not in flow_noc.\n"
    )
    file.write(
        "- Non-occluded pixels are valid in both flow_occ and flow_noc.\n"
    )
    file.write(
        "- Invalid or unlabeled pixels are excluded from evaluation.\n"
    )
    file.write(
        "- Results represent one KITTI training frame pair, "
        "not the complete benchmark.\n"
    )

print("\nSummary saved:", summary_path)


# ============================================================
# 16. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("TASK 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Results saved to:", OUTPUT_DIR)
print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(" -", path.name)
