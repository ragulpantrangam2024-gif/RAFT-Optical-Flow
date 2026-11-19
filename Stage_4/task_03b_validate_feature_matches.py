
"""
STAGE 4 - TASK 3B: GROUND-TRUTH VALIDATION OF CNN FEATURE MATCHES

Evaluates the mutual nearest-neighbour matches produced in Task 3
against KITTI ground-truth optical flow.

Metrics are calculated only at matched feature locations.
This is a sparse evaluation, not a full KITTI benchmark.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"D:\rwu\rwu jobs\Computer_Vision_Projects\RAFT_Optical_Flow"
)

DATASET_DIR = (
    PROJECT_DIR
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
)

TASK3_DIR = PROJECT_DIR / "Stage_4" / "results" / "Task 3"
OUTPUT_DIR = PROJECT_DIR / "Stage_4" / "results" / "Task 3B"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATCHES_PATH = TASK3_DIR / "feature_matches.csv"
CORRELATION_PATH = TASK3_DIR / "correlation_matrix.npy"

IMAGE_PATH = DATASET_DIR / "image_2" / "000000_10.png"
FLOW_PATH = DATASET_DIR / "flow_occ" / "000000_10.png"

# KITTI optical flow encoding constants.
FLOW_SCALE = 64.0
FLOW_OFFSET = 2 ** 15

# KITTI-style outlier thresholds.
EPE_THRESHOLD = 3.0
RELATIVE_THRESHOLD = 0.05


# ============================================================
# 2. LOAD INPUT FILES
# ============================================================

print("=" * 72)
print("STAGE 4 - TASK 3B: CNN FEATURE MATCH VALIDATION")
print("=" * 72)

for path in [MATCHES_PATH, IMAGE_PATH, FLOW_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

print("\nLoading feature matches...")
matches_df = pd.read_csv(MATCHES_PATH)

required_columns = [
    "image1_x",
    "image1_y",
    "image2_x",
    "image2_y",
    "displacement_x",
    "displacement_y",
    "cosine_similarity"
]

missing = [c for c in required_columns if c not in matches_df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

print("Matches loaded:", len(matches_df))

print("\nLoading KITTI image...")
image_bgr = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_COLOR)

if image_bgr is None:
    raise RuntimeError("Could not load KITTI image.")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

height, width = image_bgr.shape[:2]

print("Image dimensions:", width, "x", height)


# ============================================================
# 3. DECODE KITTI GROUND-TRUTH FLOW
# ============================================================

print("\nLoading KITTI ground-truth flow...")

flow_png = cv2.imread(str(FLOW_PATH), cv2.IMREAD_UNCHANGED)

if flow_png is None:
    raise RuntimeError("Could not load KITTI flow PNG.")

if flow_png.ndim != 3 or flow_png.shape[2] != 3:
    raise ValueError(
        "Unexpected KITTI flow image format. Expected 3-channel PNG."
    )

if flow_png.shape[:2] != (height, width):
    raise ValueError("Ground-truth flow and input image dimensions differ.")

# OpenCV loads the image as BGR:
# B = validity, G = encoded vertical flow, R = encoded horizontal flow.
valid_mask = flow_png[:, :, 0] > 0

gt_u = (
    flow_png[:, :, 2].astype(np.float32) - FLOW_OFFSET
) / FLOW_SCALE

gt_v = (
    flow_png[:, :, 1].astype(np.float32) - FLOW_OFFSET
) / FLOW_SCALE

ground_truth = np.stack([gt_u, gt_v], axis=-1)

print("Ground-truth valid pixels:", int(valid_mask.sum()))
print("Ground-truth invalid pixels:", int((~valid_mask).sum()))


# ============================================================
# 4. SAMPLE GROUND TRUTH AT MATCHED LOCATIONS
# ============================================================

print("\nSampling ground-truth flow at matched locations...")

source_x = matches_df["image1_x"].to_numpy(dtype=np.float32)
source_y = matches_df["image1_y"].to_numpy(dtype=np.float32)

predicted_u = matches_df["displacement_x"].to_numpy(dtype=np.float32)
predicted_v = matches_df["displacement_y"].to_numpy(dtype=np.float32)

similarity = matches_df["cosine_similarity"].to_numpy(dtype=np.float32)

# Convert image coordinates to nearest pixel.
pixel_x = np.rint(source_x).astype(int)
pixel_y = np.rint(source_y).astype(int)

inside_image = (
    (pixel_x >= 0) &
    (pixel_x < width) &
    (pixel_y >= 0) &
    (pixel_y < height)
)

valid_indices = np.where(inside_image)[0]

valid_gt_indices = valid_indices[
    valid_mask[
        pixel_y[valid_indices],
        pixel_x[valid_indices]
    ]
]

print("Total CNN matches:", len(matches_df))
print("Inside image:", len(valid_indices))
print("Matches with valid GT:", len(valid_gt_indices))

if len(valid_gt_indices) == 0:
    raise RuntimeError(
        "No matches have valid ground-truth flow. "
        "Check the KITTI flow path and validity decoding."
    )

# Keep only matches with valid ground truth.
source_x = source_x[valid_gt_indices]
source_y = source_y[valid_gt_indices]

predicted_u = predicted_u[valid_gt_indices]
predicted_v = predicted_v[valid_gt_indices]

similarity = similarity[valid_gt_indices]

pixel_x = pixel_x[valid_gt_indices]
pixel_y = pixel_y[valid_gt_indices]

gt_u_sampled = gt_u[pixel_y, pixel_x]
gt_v_sampled = gt_v[pixel_y, pixel_x]


# ============================================================
# 5. CALCULATE SPARSE MATCH ERRORS
# ============================================================

print("\nCalculating sparse optical-flow metrics...")

error_u = predicted_u - gt_u_sampled
error_v = predicted_v - gt_v_sampled

epe = np.sqrt(error_u ** 2 + error_v ** 2)

gt_magnitude = np.sqrt(
    gt_u_sampled ** 2 + gt_v_sampled ** 2
)

# KITTI-style outlier condition:
# EPE > 3 pixels AND EPE > 5% of ground-truth flow magnitude.
fl_all_mask = (
    (epe > EPE_THRESHOLD) &
    (epe > RELATIVE_THRESHOLD * gt_magnitude)
)

fl_all = 100.0 * np.mean(fl_all_mask)

mean_epe = float(np.mean(epe))
median_epe = float(np.median(epe))
std_epe = float(np.std(epe))
max_epe = float(np.max(epe))

mean_gt_magnitude = float(np.mean(gt_magnitude))

below_1 = 100.0 * np.mean(epe < 1.0)
below_3 = 100.0 * np.mean(epe < 3.0)
below_5 = 100.0 * np.mean(epe < 5.0)

mean_similarity = float(np.mean(similarity))

print("\nSPARSE MATCH VALIDATION RESULTS")
print("-" * 72)
print("Evaluated matches:", len(epe))
print(f"Mean EPE: {mean_epe:.6f} pixels")
print(f"Median EPE: {median_epe:.6f} pixels")
print(f"Standard deviation: {std_epe:.6f} pixels")
print(f"Maximum EPE: {max_epe:.6f} pixels")
print(f"Mean GT motion magnitude: {mean_gt_magnitude:.6f} pixels")
print(f"Fl-all: {fl_all:.4f}%")
print(f"EPE below 1 pixel: {below_1:.4f}%")
print(f"EPE below 3 pixels: {below_3:.4f}%")
print(f"EPE below 5 pixels: {below_5:.4f}%")
print(f"Mean cosine similarity: {mean_similarity:.6f}")


# ============================================================
# 6. CREATE MATCH-LEVEL RESULTS CSV
# ============================================================

results_df = pd.DataFrame({
    "image1_x": source_x,
    "image1_y": source_y,
    "predicted_u": predicted_u,
    "predicted_v": predicted_v,
    "ground_truth_u": gt_u_sampled,
    "ground_truth_v": gt_v_sampled,
    "error_u": error_u,
    "error_v": error_v,
    "endpoint_error": epe,
    "ground_truth_magnitude": gt_magnitude,
    "cosine_similarity": similarity,
    "is_fl_all_outlier": fl_all_mask
})

results_df.to_csv(
    OUTPUT_DIR / "sparse_match_validation.csv",
    index=False
)

print("\nSaved sparse_match_validation.csv")


# ============================================================
# 7. CORRELATION MATRIX ANALYSIS
# ============================================================

if CORRELATION_PATH.exists():

    print("\nInspecting saved correlation matrix...")

    correlation = np.load(CORRELATION_PATH)

    correlation_stats = {
        "minimum": float(np.min(correlation)),
        "maximum": float(np.max(correlation)),
        "mean": float(np.mean(correlation)),
        "standard_deviation": float(np.std(correlation))
    }

    print("Correlation matrix shape:", correlation.shape)

    for name, value in correlation_stats.items():
        print(f"{name}: {value:.8f}")

else:
    correlation_stats = None
    print("\nCorrelation matrix not found; skipping matrix statistics.")


# ============================================================
# 8. VISUALIZATION: EPE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(epe, bins=35, edgecolor="black")

plt.xlabel("Endpoint Error (pixels)")
plt.ylabel("Number of Matches")
plt.title("Sparse CNN Feature Match EPE Distribution")
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "sparse_match_epe_distribution.png",
    dpi=200
)
plt.close()


# ============================================================
# 9. VISUALIZATION: SIMILARITY VS EPE
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    similarity,
    epe,
    alpha=0.7
)

plt.xlabel("CNN Cosine Similarity")
plt.ylabel("Endpoint Error (pixels)")
plt.title("Feature Similarity vs Ground-Truth Match Error")
plt.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "similarity_vs_epe.png",
    dpi=200
)
plt.close()


# ============================================================
# 10. VISUALIZATION: PREDICTED VS GROUND-TRUTH FLOW
# ============================================================

# Draw only a subset to keep the visualization readable.
MAX_ARROWS = 150
step = max(1, len(epe) // MAX_ARROWS)
selected = np.arange(0, len(epe), step)[:MAX_ARROWS]

plt.figure(figsize=(16, 7))
plt.imshow(image_rgb)

# Green: match passes KITTI-style outlier test.
# Red: match is a KITTI-style outlier.
for index in selected:

    x = source_x[index]
    y = source_y[index]

    color = "red" if fl_all_mask[index] else "lime"

    plt.arrow(
        x, y,
        predicted_u[index],
        predicted_v[index],
        color=color,
        width=0.35,
        head_width=4,
        length_includes_head=True
    )

plt.title(
    "CNN Feature Displacements: Green = Inlier, Red = Fl-all Outlier"
)
plt.axis("off")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "validated_feature_displacements.png",
    dpi=200
)
plt.close()


# ============================================================
# 11. SAVE SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "task_03b_validation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 4 - TASK 3B: GROUND-TRUTH VALIDATION\n")
    file.write("=" * 65 + "\n\n")

    file.write("Dataset: KITTI Scene Flow training\n")
    file.write("Image pair: 000000_10.png -> 000000_11.png\n")
    file.write(f"Image dimensions: {width} x {height}\n\n")

    file.write(f"Original mutual matches: {len(matches_df)}\n")
    file.write(f"Matches inside image: {len(valid_indices)}\n")
    file.write(f"Matches evaluated with valid GT: {len(epe)}\n\n")

    file.write(f"Mean EPE: {mean_epe:.6f} pixels\n")
    file.write(f"Median EPE: {median_epe:.6f} pixels\n")
    file.write(f"Standard deviation: {std_epe:.6f} pixels\n")
    file.write(f"Maximum EPE: {max_epe:.6f} pixels\n")
    file.write(f"Fl-all: {fl_all:.6f}%\n")

    file.write(f"EPE below 1 pixel: {below_1:.6f}%\n")
    file.write(f"EPE below 3 pixels: {below_3:.6f}%\n")
    file.write(f"EPE below 5 pixels: {below_5:.6f}%\n")

    file.write(
        f"\nMean cosine similarity: {mean_similarity:.8f}\n"
    )

    if correlation_stats is not None:
        file.write("\nCORRELATION MATRIX\n")
        for name, value in correlation_stats.items():
            file.write(f"{name}: {value:.8f}\n")

    file.write("\nINTERPRETATION\n")
    file.write(
        "These metrics evaluate only the sparse mutual nearest-neighbour "
        "matches generated by Task 3. They are not dense KITTI benchmark "
        "metrics. The CNN uses randomly initialized weights, so the "
        "matches are not learned optical-flow correspondences.\n"
    )

print("\nSaved task_03b_validation_summary.txt")


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 72)
print("TASK 3B COMPLETED SUCCESSFULLY")
print("=" * 72)

print("Results directory:", OUTPUT_DIR)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)
    