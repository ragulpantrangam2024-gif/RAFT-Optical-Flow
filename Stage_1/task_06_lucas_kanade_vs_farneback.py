
"""
STAGE 1 - TASK 6
COMPARATIVE EVALUATION:
LUCAS-KANADE VS FARNEBACK

Dataset: MPI Sintel
Sequence: alley_1
Frame pair: frame_0001 -> frame_0002

Comparison:
    1. Lucas-Kanade sparse optical flow
    2. Farneback dense optical flow

Fair comparison:
    Both algorithms are evaluated at the same Lucas-Kanade
    feature locations using bilinearly sampled ground truth.

Additional evaluation:
    Farneback is also evaluated across the complete valid image.

Metrics:
    - Mean EPE
    - Median EPE
    - EPE below 1, 3 and 5 pixels
    - Mean Angular Error
"""

from pathlib import Path
import csv
import struct

import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LK_RESULTS_PATH = (
    PROJECT_ROOT
    / "Stage_1"
    / "results"
    / "Task 3"
    / "optical_flow_vectors.csv"
)

FARNEBACK_RESULTS_PATH = (
    PROJECT_ROOT
    / "Stage_1"
    / "results"
    / "Task 4"
    / "farneback_flow.npy"
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "Stage_1"
    / "results"
    / "Task 6"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. READ MIDDLEBURY .FLO FILE
# --------------------------------------------------

def read_flo_file(path):

    with open(path, "rb") as file:

        magic = struct.unpack("<f", file.read(4))[0]

        if magic != 202021.25:
            raise ValueError("Invalid .flo file.")

        width = struct.unpack("<i", file.read(4))[0]
        height = struct.unpack("<i", file.read(4))[0]

        data = np.fromfile(file, dtype="<f4")

    expected_size = width * height * 2

    if data.size != expected_size:
        raise ValueError("Unexpected ground-truth flow data size.")

    return data.reshape(height, width, 2)


# --------------------------------------------------
# 3. LOAD INPUT DATA
# --------------------------------------------------

for path in [
    LK_RESULTS_PATH,
    FARNEBACK_RESULTS_PATH,
    GROUND_TRUTH_PATH,
    INVALID_MASK_PATH
]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

lk_data = np.genfromtxt(
    LK_RESULTS_PATH,
    delimiter=",",
    names=True
)

if lk_data.size == 0:
    raise ValueError("Lucas-Kanade CSV contains no tracking results.")

lk_data = np.atleast_1d(lk_data)

# Extract feature coordinates and LK flow.
x = lk_data["x1"].astype(np.float32)
y = lk_data["y1"].astype(np.float32)

lk_u = lk_data["u"].astype(np.float32)
lk_v = lk_data["v"].astype(np.float32)

lk_flow = np.column_stack((lk_u, lk_v))

farneback_flow = np.load(FARNEBACK_RESULTS_PATH)

ground_truth_flow = read_flo_file(GROUND_TRUTH_PATH)

height, width, _ = ground_truth_flow.shape

if farneback_flow.shape != ground_truth_flow.shape:
    raise ValueError("Farneback and ground-truth dimensions differ.")

if farneback_flow.shape[2] != 2:
    raise ValueError("Farneback flow must have two components.")

print("=" * 65)
print("STAGE 1 - TASK 6")
print("LUCAS-KANADE VS FARNEBACK")
print("=" * 65)

print(f"\nImage dimensions: {width} x {height}")
print(f"Lucas-Kanade feature points: {len(x)}")


# --------------------------------------------------
# 4. LOAD INVALID PIXEL MASK
# --------------------------------------------------

invalid_mask = cv2.imread(
    str(INVALID_MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if invalid_mask is None:
    raise ValueError("Could not read invalid-pixel mask.")

if invalid_mask.shape != (height, width):
    raise ValueError("Invalid mask dimensions do not match.")

valid_ground_truth = invalid_mask == 0

# Sample mask at feature locations using nearest-neighbour.
map_x = x.reshape(1, -1)
map_y = y.reshape(1, -1)

sampled_invalid_mask = cv2.remap(
    invalid_mask,
    map_x,
    map_y,
    interpolation=cv2.INTER_NEAREST,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=255
).reshape(-1)

# Feature locations must be inside the image.
inside_image = (
    (x >= 0)
    & (x <= width - 1)
    & (y >= 0)
    & (y <= height - 1)
)

valid_features = (
    inside_image
    & (sampled_invalid_mask == 0)
    & np.isfinite(lk_flow).all(axis=1)
)

print(f"Valid feature locations: {np.count_nonzero(valid_features)}")
print(f"Excluded feature locations: {np.count_nonzero(~valid_features)}")


# --------------------------------------------------
# 5. BILINEAR SAMPLING OF GROUND TRUTH
# --------------------------------------------------

# OpenCV remap uses bilinear interpolation by default.
# This gives ground-truth flow at subpixel feature locations.

gt_u = cv2.remap(
    ground_truth_flow[..., 0],
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=np.nan
).reshape(-1)

gt_v = cv2.remap(
    ground_truth_flow[..., 1],
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=np.nan
).reshape(-1)

gt_sampled = np.column_stack((gt_u, gt_v))

valid_features &= np.isfinite(gt_sampled).all(axis=1)


# --------------------------------------------------
# 6. SAMPLE FARNEBACK AT SAME FEATURE LOCATIONS
# --------------------------------------------------

fb_u = cv2.remap(
    farneback_flow[..., 0],
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=np.nan
).reshape(-1)

fb_v = cv2.remap(
    farneback_flow[..., 1],
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=np.nan
).reshape(-1)

fb_sampled = np.column_stack((fb_u, fb_v))

valid_features &= np.isfinite(fb_sampled).all(axis=1)

if not np.any(valid_features):
    raise ValueError("No valid common feature locations for comparison.")

lk_valid = lk_flow[valid_features]
fb_valid = fb_sampled[valid_features]
gt_valid = gt_sampled[valid_features]

print(f"Common evaluation locations: {len(gt_valid)}")


# --------------------------------------------------
# 7. EVALUATION FUNCTIONS
# --------------------------------------------------

def calculate_epe(estimated, ground_truth):

    difference = estimated - ground_truth

    return np.sqrt(
        difference[:, 0] ** 2
        + difference[:, 1] ** 2
    )


def calculate_angular_error(estimated, ground_truth):

    estimated_magnitude = np.linalg.norm(
        estimated,
        axis=1
    )

    ground_truth_magnitude = np.linalg.norm(
        ground_truth,
        axis=1
    )

    valid_direction = (
        (estimated_magnitude > 1e-6)
        & (ground_truth_magnitude > 1e-6)
    )

    if not np.any(valid_direction):
        return float("nan")

    estimated_direction = estimated[valid_direction]
    ground_truth_direction = ground_truth[valid_direction]

    dot = np.sum(
        estimated_direction * ground_truth_direction,
        axis=1
    )

    denominator = (
        estimated_magnitude[valid_direction]
        * ground_truth_magnitude[valid_direction]
    )

    cosine = np.clip(dot / denominator, -1.0, 1.0)

    angles = np.degrees(np.arccos(cosine))

    return float(np.mean(angles))


def calculate_metrics(estimated, ground_truth):

    epe = calculate_epe(estimated, ground_truth)

    return {
        "Mean EPE": float(np.mean(epe)),
        "Median EPE": float(np.median(epe)),
        "Maximum EPE": float(np.max(epe)),
        "EPE < 1 px (%)": float(np.mean(epe < 1) * 100),
        "EPE < 3 px (%)": float(np.mean(epe < 3) * 100),
        "EPE < 5 px (%)": float(np.mean(epe < 5) * 100),
        "Mean Angular Error": calculate_angular_error(
            estimated,
            ground_truth
        ),
        "Evaluation points": len(epe)
    }


# --------------------------------------------------
# 8. COMPARE BOTH METHODS AT SAME POINTS
# --------------------------------------------------

lk_metrics = calculate_metrics(
    lk_valid,
    gt_valid
)

fb_metrics = calculate_metrics(
    fb_valid,
    gt_valid
)

lk_epe = calculate_epe(lk_valid, gt_valid)
fb_epe = calculate_epe(fb_valid, gt_valid)


# --------------------------------------------------
# 9. FULL-IMAGE FARNEBACK EVALUATION
# --------------------------------------------------

finite_mask = (
    np.isfinite(ground_truth_flow).all(axis=2)
    & np.isfinite(farneback_flow).all(axis=2)
)

full_valid_mask = valid_ground_truth & finite_mask

full_epe = np.sqrt(
    np.sum(
        (farneback_flow - ground_truth_flow) ** 2,
        axis=2
    )
)

full_valid_epe = full_epe[full_valid_mask]

fb_full_metrics = {
    "Mean EPE": float(np.mean(full_valid_epe)),
    "Median EPE": float(np.median(full_valid_epe)),
    "Maximum EPE": float(np.max(full_valid_epe)),
    "EPE < 1 px (%)": float(np.mean(full_valid_epe < 1) * 100),
    "EPE < 3 px (%)": float(np.mean(full_valid_epe < 3) * 100),
    "EPE < 5 px (%)": float(np.mean(full_valid_epe < 5) * 100),
    "Evaluation points": len(full_valid_epe)
}


# --------------------------------------------------
# 10. PRINT COMPARISON RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("FAIR COMPARISON - SAME FEATURE LOCATIONS")
print("=" * 65)

print(f"\nEvaluation points: {len(gt_valid)}")

for metric in [
    "Mean EPE",
    "Median EPE",
    "Maximum EPE",
    "EPE < 1 px (%)",
    "EPE < 3 px (%)",
    "EPE < 5 px (%)",
    "Mean Angular Error"
]:

    print(f"\n{metric}")
    print(f"  Lucas-Kanade: {lk_metrics[metric]:.6f}")
    print(f"  Farneback:    {fb_metrics[metric]:.6f}")


print("\n" + "=" * 65)
print("FULL-IMAGE FARNEBACK EVALUATION")
print("=" * 65)

print(f"Valid pixels: {fb_full_metrics['Evaluation points']}")
print(f"Mean EPE: {fb_full_metrics['Mean EPE']:.6f}")
print(f"Median EPE: {fb_full_metrics['Median EPE']:.6f}")
print(f"EPE < 1 px: {fb_full_metrics['EPE < 1 px (%)']:.2f}%")
print(f"EPE < 3 px: {fb_full_metrics['EPE < 3 px (%)']:.2f}%")
print(f"EPE < 5 px: {fb_full_metrics['EPE < 5 px (%)']:.2f}%")


# --------------------------------------------------
# 11. SAVE POINT-WISE COMPARISON
# --------------------------------------------------

point_results = np.column_stack((
    x[valid_features],
    y[valid_features],
    gt_valid[:, 0],
    gt_valid[:, 1],
    lk_valid[:, 0],
    lk_valid[:, 1],
    lk_epe,
    fb_valid[:, 0],
    fb_valid[:, 1],
    fb_epe
))

np.savetxt(
    RESULTS_DIR / "flow_comparison_at_lk_points.csv",
    point_results,
    delimiter=",",
    header=(
        "x,y,gt_u,gt_v,lk_u,lk_v,lk_epe,"
        "fb_u,fb_v,fb_epe"
    ),
    comments="",
    fmt="%.6f"
)


# --------------------------------------------------
# 12. SAVE METRICS CSV
# --------------------------------------------------

metrics_path = RESULTS_DIR / "comparison_metrics.csv"

with open(
    metrics_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow(["Metric", "Lucas-Kanade", "Farneback (same points)"])

    for metric in [
        "Mean EPE",
        "Median EPE",
        "Maximum EPE",
        "EPE < 1 px (%)",
        "EPE < 3 px (%)",
        "EPE < 5 px (%)",
        "Mean Angular Error"
    ]:

        writer.writerow([
            metric,
            lk_metrics[metric],
            fb_metrics[metric]
        ])

    writer.writerow([])
    writer.writerow(["Farneback full-image evaluation", "Value"])

    for metric, value in fb_full_metrics.items():
        writer.writerow([metric, value])


# --------------------------------------------------
# 13. CREATE EPE COMPARISON CHART
# --------------------------------------------------

plt.figure(figsize=(9, 6))

methods = ["Lucas-Kanade", "Farneback"]

mean_errors = [
    lk_metrics["Mean EPE"],
    fb_metrics["Mean EPE"]
]

plt.bar(methods, mean_errors)

plt.ylabel("Mean Endpoint Error (pixels)")
plt.title("Lucas-Kanade vs Farneback - Same Feature Locations")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "mean_epe_comparison.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 14. CREATE EPE DISTRIBUTION COMPARISON
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.hist(
    lk_epe,
    bins=40,
    alpha=0.6,
    label="Lucas-Kanade"
)

plt.hist(
    fb_epe,
    bins=40,
    alpha=0.6,
    label="Farneback"
)

plt.xlabel("Endpoint Error (pixels)")
plt.ylabel("Number of Feature Points")
plt.title("EPE Distribution - Same Feature Locations")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "epe_distribution_comparison.png",
    dpi=150
)

plt.close()


# --------------------------------------------------
# 15. SAVE TEXT SUMMARY
# --------------------------------------------------

summary = f"""
STAGE 1 - TASK 6
LUCAS-KANADE VS FARNEBACK

Dataset: MPI Sintel
Sequence: alley_1
Frame pair: frame_0001 -> frame_0002

Image dimensions: {width} x {height}

FAIR COMPARISON
Both algorithms are evaluated at the same
Lucas-Kanade feature locations.

Valid common evaluation points: {len(gt_valid)}

LUCAS-KANADE:
    Mean EPE: {lk_metrics['Mean EPE']:.6f}
    Median EPE: {lk_metrics['Median EPE']:.6f}
    Maximum EPE: {lk_metrics['Maximum EPE']:.6f}
    EPE < 1 px: {lk_metrics['EPE < 1 px (%)']:.4f}%
    EPE < 3 px: {lk_metrics['EPE < 3 px (%)']:.4f}%
    EPE < 5 px: {lk_metrics['EPE < 5 px (%)']:.4f}%
    Mean Angular Error: {lk_metrics['Mean Angular Error']:.6f} degrees

FARNEBACK AT SAME FEATURE LOCATIONS:
    Mean EPE: {fb_metrics['Mean EPE']:.6f}
    Median EPE: {fb_metrics['Median EPE']:.6f}
    Maximum EPE: {fb_metrics['Maximum EPE']:.6f}
    EPE < 1 px: {fb_metrics['EPE < 1 px (%)']:.4f}%
    EPE < 3 px: {fb_metrics['EPE < 3 px (%)']:.4f}%
    EPE < 5 px: {fb_metrics['EPE < 5 px (%)']:.4f}%
    Mean Angular Error: {fb_metrics['Mean Angular Error']:.6f} degrees

FARNEBACK FULL-IMAGE EVALUATION:
    Valid pixels: {fb_full_metrics['Evaluation points']}
    Mean EPE: {fb_full_metrics['Mean EPE']:.6f}
    Median EPE: {fb_full_metrics['Median EPE']:.6f}
    EPE < 1 px: {fb_full_metrics['EPE < 1 px (%)']:.4f}%
    EPE < 3 px: {fb_full_metrics['EPE < 3 px (%)']:.4f}%
    EPE < 5 px: {fb_full_metrics['EPE < 5 px (%)']:.4f}%

Methodological note:
    Farneback was bilinearly sampled at the original
    Lucas-Kanade feature coordinates.
    Ground truth was also bilinearly sampled there.
    Invalid ground-truth locations were excluded.
    Full-image Farneback metrics use all valid pixels.

The two full-image and feature-point evaluations
represent different evaluation populations.
"""

(RESULTS_DIR / "comparison_summary.txt").write_text(
    summary,
    encoding="utf-8"
)


# --------------------------------------------------
# 16. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TASK 6 COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"\nResults saved in:\n{RESULTS_DIR}")

print("\nGenerated files:")
print("  - flow_comparison_at_lk_points.csv")
print("  - comparison_metrics.csv")
print("  - mean_epe_comparison.png")
print("  - epe_distribution_comparison.png")
print("  - comparison_summary.txt")
