
"""
STAGE 3 - TASK 4: MOTION-BASED ERROR ANALYSIS

Analyses optical flow accuracy at different ground-truth motion magnitudes
using the KITTI 2015 training dataset.

Motion categories:
    Small:  0 <= magnitude < 5 pixels
    Medium: 5 <= magnitude < 20 pixels
    Large:  magnitude >= 20 pixels

Algorithms:
    1. Farneback
    2. Horn-Schunck
    3. TV-L1
    4. DIS Medium

Outputs:
    Stage_3/results/Task 4/
"""

from pathlib import Path
import time
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\rwu\rwu jobs\Computer_Vision_Projects\RAFT_Optical_Flow"
)

KITTI_ROOT = (
    PROJECT_ROOT
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
)

IMAGE_1_PATH = KITTI_ROOT / "image_2" / "000000_10.png"
IMAGE_2_PATH = KITTI_ROOT / "image_2" / "000000_11.png"
FLOW_GT_PATH = KITTI_ROOT / "flow_occ" / "000000_10.png"

OUTPUT_DIR = PROJECT_ROOT / "Stage_3" / "results" / "Task 4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best tested configurations from Task 3
FARNEBACK_WINSIZE = 15
HORN_SCHUNCK_ALPHA = 0.5
TVL1_LAMBDA = 0.10
DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM

HS_ITERATIONS = 100

MOTION_BINS = {
    "Small (0-5 px)": (0.0, 5.0),
    "Medium (5-20 px)": (5.0, 20.0),
    "Large (>=20 px)": (20.0, np.inf),
}


# ============================================================
# 2. LOAD KITTI IMAGES
# ============================================================

def load_images():

    print("\nLoading KITTI images...")

    image_1 = cv2.imread(str(IMAGE_1_PATH))
    image_2 = cv2.imread(str(IMAGE_2_PATH))

    if image_1 is None or image_2 is None:
        raise FileNotFoundError(
            "Could not load KITTI images. Check the image paths."
        )

    if image_1.shape != image_2.shape:
        raise ValueError("The two input images have different dimensions.")

    gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    print("Image shape:", gray_1.shape)

    return image_1, gray_1, gray_2


# ============================================================
# 3. LOAD AND DECODE KITTI GROUND-TRUTH FLOW
# ============================================================

def load_kitti_ground_truth():

    print("\nLoading KITTI ground-truth flow...")

    flow_png = cv2.imread(
        str(FLOW_GT_PATH),
        cv2.IMREAD_UNCHANGED
    )

    if flow_png is None:
        raise FileNotFoundError(
            "Could not load KITTI ground-truth flow."
        )

    if flow_png.ndim != 3 or flow_png.shape[2] != 3:
        raise ValueError("Unexpected KITTI flow PNG format.")

    # KITTI flow encoding:
    # R = horizontal flow, G = vertical flow, B = validity
    # OpenCV loads PNG channels in BGR order.
    flow_u = (flow_png[:, :, 2].astype(np.float32) - 32768.0) / 64.0
    flow_v = (flow_png[:, :, 1].astype(np.float32) - 32768.0) / 64.0

    ground_truth = np.dstack((flow_u, flow_v))

    # KITTI validity channel: blue channel > 0
    valid_mask = flow_png[:, :, 0] > 0

    if not np.any(valid_mask):
        raise ValueError("No valid ground-truth pixels were found.")

    print("Ground-truth shape:", ground_truth.shape)
    print("Valid pixels:", int(np.count_nonzero(valid_mask)))

    return ground_truth, valid_mask


# ============================================================
# 4. HORN-SCHUNCK IMPLEMENTATION
# ============================================================

def horn_schunck(image_1, image_2, alpha, iterations):

    first = image_1.astype(np.float32) / 255.0
    second = image_2.astype(np.float32) / 255.0

    # Spatial and temporal image derivatives
    Ix = cv2.Sobel(
        first, cv2.CV_32F, 1, 0, ksize=3
    ) / 8.0

    Iy = cv2.Sobel(
        first, cv2.CV_32F, 0, 1, ksize=3
    ) / 8.0

    It = second - first

    u = np.zeros_like(first, dtype=np.float32)
    v = np.zeros_like(first, dtype=np.float32)

    averaging_kernel = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6,  0.0,  1 / 6],
            [1 / 12, 1 / 6, 1 / 12],
        ],
        dtype=np.float32
    )

    alpha_squared = alpha ** 2

    for _ in range(iterations):

        u_avg = cv2.filter2D(
            u, -1, averaging_kernel,
            borderType=cv2.BORDER_REFLECT
        )

        v_avg = cv2.filter2D(
            v, -1, averaging_kernel,
            borderType=cv2.BORDER_REFLECT
        )

        common = (
            Ix * u_avg
            + Iy * v_avg
            + It
        ) / (
            alpha_squared + Ix ** 2 + Iy ** 2
        )

        u = u_avg - Ix * common
        v = v_avg - Iy * common

    return np.dstack((u, v)).astype(np.float32)


# ============================================================
# 5. COMPUTE OPTICAL FLOW USING FOUR ALGORITHMS
# ============================================================

def calculate_optical_flows(gray_1, gray_2):

    flows = {}
    runtimes = {}

    print("\nCalculating optical flow...")

    # --------------------------------------------------------
    # Farneback
    # --------------------------------------------------------

    print("\nRunning Farneback...")

    start = time.perf_counter()

    farneback = cv2.calcOpticalFlowFarneback(
        gray_1,
        gray_2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=FARNEBACK_WINSIZE,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    runtimes["Farneback"] = time.perf_counter() - start
    flows["Farneback"] = farneback

    # --------------------------------------------------------
    # Horn-Schunck
    # --------------------------------------------------------

    print("Running Horn-Schunck...")

    start = time.perf_counter()

    hs_flow = horn_schunck(
        gray_1,
        gray_2,
        alpha=HORN_SCHUNCK_ALPHA,
        iterations=HS_ITERATIONS
    )

    runtimes["Horn-Schunck"] = time.perf_counter() - start
    flows["Horn-Schunck"] = hs_flow

    # --------------------------------------------------------
    # TV-L1
    # --------------------------------------------------------

    print("Running TV-L1...")

    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            "OpenCV optflow module is missing. "
            "Install opencv-contrib-python in your environment."
        )

    start = time.perf_counter()

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    tvl1.setLambda(TVL1_LAMBDA)

    tvl1_flow = tvl1.calc(gray_1, gray_2, None)

    runtimes["TV-L1"] = time.perf_counter() - start
    flows["TV-L1"] = tvl1_flow

    # --------------------------------------------------------
    # DIS MEDIUM
    # --------------------------------------------------------

    print("Running DIS Medium...")

    start = time.perf_counter()

    dis = cv2.DISOpticalFlow_create(DIS_PRESET)

    dis_flow = dis.calc(gray_1, gray_2, None)

    runtimes["DIS Medium"] = time.perf_counter() - start
    flows["DIS Medium"] = dis_flow

    return flows, runtimes


# ============================================================
# 6. CALCULATE GROUND-TRUTH MOTION MAGNITUDE
# ============================================================

def calculate_ground_truth_magnitude(ground_truth):

    gt_u = ground_truth[:, :, 0]
    gt_v = ground_truth[:, :, 1]

    return np.sqrt(gt_u ** 2 + gt_v ** 2)


# ============================================================
# 7. EVALUATE EACH MOTION CATEGORY
# ============================================================

def evaluate_motion_categories(
    flows,
    ground_truth,
    valid_mask,
    gt_magnitude
):

    results = []

    print("\nStarting motion-based evaluation")
    print("=" * 75)

    for algorithm_name, predicted_flow in flows.items():

        print(f"\nAlgorithm: {algorithm_name}")

        if predicted_flow.shape != ground_truth.shape:
            raise ValueError(
                f"Flow shape mismatch for {algorithm_name}: "
                f"{predicted_flow.shape} vs {ground_truth.shape}"
            )

        # Endpoint error for every pixel
        error = np.linalg.norm(
            predicted_flow - ground_truth,
            axis=2
        )

        # KITTI Fl-all outlier definition:
        # EPE > 3 pixels AND relative EPE > 5%
        relative_error = error / np.maximum(
            gt_magnitude, 1e-6
        )

        outlier_mask = (
            (error > 3.0)
            & (relative_error > 0.05)
        )

        for category, (lower, upper) in MOTION_BINS.items():

            category_mask = (
                valid_mask
                & (gt_magnitude >= lower)
                & (gt_magnitude < upper)
            )

            count = int(np.count_nonzero(category_mask))

            if count == 0:
                print(f"  {category}: no valid pixels")
                continue

            category_errors = error[category_mask]
            category_outliers = outlier_mask[category_mask]

            mean_epe = float(np.mean(category_errors))
            median_epe = float(np.median(category_errors))
            std_epe = float(np.std(category_errors))

            fl_all = float(
                np.mean(category_outliers) * 100.0
            )

            epe_below_3 = float(
                np.mean(category_errors < 3.0) * 100.0
            )

            result = {
                "Algorithm": algorithm_name,
                "Motion Category": category,
                "Valid Pixels": count,
                "Mean EPE (px)": mean_epe,
                "Median EPE (px)": median_epe,
                "Std EPE (px)": std_epe,
                "Fl-all (%)": fl_all,
                "EPE Below 3px (%)": epe_below_3,
            }

            results.append(result)

            print(f"\n  {category}")
            print(f"  Valid pixels: {count}")
            print(f"  Mean EPE: {mean_epe:.6f}")
            print(f"  Median EPE: {median_epe:.6f}")
            print(f"  Fl-all: {fl_all:.6f}%")
            print(f"  EPE below 3px: {epe_below_3:.6f}%")

    return results


# ============================================================
# 8. SAVE RESULTS TO CSV
# ============================================================

def save_results_csv(results):

    csv_path = OUTPUT_DIR / "motion_based_error_metrics.csv"

    if not results:
        raise ValueError("No evaluation results to save.")

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nCSV saved:", csv_path)


# ============================================================
# 9. GENERATE MEAN EPE COMPARISON PLOT
# ============================================================

def plot_mean_epe(results):

    categories = list(MOTION_BINS.keys())
    algorithms = list(dict.fromkeys(
        row["Algorithm"] for row in results
    ))

    x = np.arange(len(categories))
    width = 0.19

    plt.figure(figsize=(12, 7))

    for index, algorithm in enumerate(algorithms):

        values = []

        for category in categories:

            matching = [
                row for row in results
                if row["Algorithm"] == algorithm
                and row["Motion Category"] == category
            ]

            values.append(
                matching[0]["Mean EPE (px)"]
                if matching else np.nan
            )

        offset = (index - (len(algorithms) - 1) / 2) * width

        plt.bar(
            x + offset,
            values,
            width,
            label=algorithm
        )

    plt.xticks(x, categories)
    plt.xlabel("Ground-Truth Motion Magnitude")
    plt.ylabel("Mean Endpoint Error (pixels)")
    plt.title("Mean EPE Across Motion Categories")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "mean_epe_by_motion_category.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


# ============================================================
# 10. GENERATE FL-ALL COMPARISON PLOT
# ============================================================

def plot_fl_all(results):

    categories = list(MOTION_BINS.keys())
    algorithms = list(dict.fromkeys(
        row["Algorithm"] for row in results
    ))

    x = np.arange(len(categories))
    width = 0.19

    plt.figure(figsize=(12, 7))

    for index, algorithm in enumerate(algorithms):

        values = []

        for category in categories:

            matching = [
                row for row in results
                if row["Algorithm"] == algorithm
                and row["Motion Category"] == category
            ]

            values.append(
                matching[0]["Fl-all (%)"]
                if matching else np.nan
            )

        offset = (index - (len(algorithms) - 1) / 2) * width

        plt.bar(
            x + offset,
            values,
            width,
            label=algorithm
        )

    plt.xticks(x, categories)
    plt.xlabel("Ground-Truth Motion Magnitude")
    plt.ylabel("KITTI Fl-all Outliers (%)")
    plt.title("Fl-all Across Motion Categories")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "fl_all_by_motion_category.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


# ============================================================
# 11. GENERATE EPE DISTRIBUTION PLOT
# ============================================================

def plot_epe_distribution(
    flows,
    ground_truth,
    valid_mask,
    gt_magnitude
):

    categories = list(MOTION_BINS.keys())

    fig, axes = plt.subplots(
        1, 3,
        figsize=(17, 6)
    )

    for axis, category in zip(axes, categories):

        lower, upper = MOTION_BINS[category]

        category_mask = (
            valid_mask
            & (gt_magnitude >= lower)
            & (gt_magnitude < upper)
        )

        for algorithm, predicted_flow in flows.items():

            error = np.linalg.norm(
                predicted_flow - ground_truth,
                axis=2
            )

            values = error[category_mask]

            # Limit histogram range for readable visualization.
            # Metrics in the CSV still use all valid category pixels.
            axis.hist(
                values,
                bins=60,
                range=(0, 50),
                alpha=0.4,
                label=algorithm,
                density=True
            )

        axis.set_title(category)
        axis.set_xlabel("Endpoint Error (pixels)")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.3)

    axes[-1].legend(fontsize=8)

    fig.suptitle("Endpoint Error Distribution by Motion Category")
    fig.tight_layout()

    path = OUTPUT_DIR / "epe_distribution_by_motion.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)

    print("Saved:", path)


# ============================================================
# 12. GENERATE MOTION MAGNITUDE MAP
# ============================================================

def save_motion_magnitude_map(gt_magnitude, valid_mask):

    display = np.full(
        gt_magnitude.shape,
        np.nan,
        dtype=np.float32
    )

    display[valid_mask] = gt_magnitude[valid_mask]

    plt.figure(figsize=(14, 6))

    image = plt.imshow(
        display,
        cmap="turbo",
        vmin=0,
        vmax=np.percentile(gt_magnitude[valid_mask], 99)
    )

    plt.title("KITTI Ground-Truth Motion Magnitude")
    plt.axis("off")
    plt.colorbar(
        image,
        label="Motion Magnitude (pixels)"
    )

    plt.tight_layout()

    path = OUTPUT_DIR / "ground_truth_motion_magnitude.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print("Saved:", path)


# ============================================================
# 13. SAVE TEXT SUMMARY
# ============================================================

def save_summary(results, runtimes, valid_mask, gt_magnitude):

    summary_path = OUTPUT_DIR / "motion_analysis_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STAGE 3 - TASK 4: MOTION-BASED ERROR ANALYSIS\n"
        )
        file.write("=" * 65 + "\n\n")

        file.write("Dataset: KITTI 2015 training\n")
        file.write("Sample: 000000_10.png -> 000000_11.png\n")
        file.write(
            f"Total valid pixels: {np.count_nonzero(valid_mask)}\n"
        )

        file.write(
            f"Mean ground-truth motion magnitude: "
            f"{np.mean(gt_magnitude[valid_mask]):.6f} px\n\n"
        )

        file.write("Algorithm runtimes:\n")

        for algorithm, runtime in runtimes.items():
            file.write(
                f"{algorithm}: {runtime:.6f} seconds\n"
            )

        file.write("\nMotion category results:\n")
        file.write("=" * 65 + "\n")

        for row in results:

            file.write(
                f"\n{row['Algorithm']} - "
                f"{row['Motion Category']}\n"
            )

            for key, value in row.items():

                if key in ("Algorithm", "Motion Category"):
                    continue

                file.write(f"{key}: {value}\n")

        file.write("\nImportant note:\n")
        file.write(
            "Results are based on one KITTI training frame pair. "
            "They are not full-dataset benchmark results.\n"
        )

        file.write(
            "Motion categories are based on ground-truth flow "
            "magnitude, not predicted flow magnitude.\n"
        )

    print("Summary saved:", summary_path)


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STAGE 3 - TASK 4: MOTION-BASED ERROR ANALYSIS")
    print("=" * 75)

    # Load data
    image_1, gray_1, gray_2 = load_images()

    ground_truth, valid_mask = load_kitti_ground_truth()

    if gray_1.shape != ground_truth.shape[:2]:
        raise ValueError(
            "Image and ground-truth dimensions do not match."
        )

    # Calculate ground-truth motion magnitude
    gt_magnitude = calculate_ground_truth_magnitude(
        ground_truth
    )

    # Display motion category pixel counts
    print("\nMotion category pixel counts:")
    for category, (lower, upper) in MOTION_BINS.items():

        mask = (
            valid_mask
            & (gt_magnitude >= lower)
            & (gt_magnitude < upper)
        )

        print(f"{category}: {np.count_nonzero(mask)}")

    # Calculate flow
    flows, runtimes = calculate_optical_flows(
        gray_1,
        gray_2
    )

    # Evaluate
    results = evaluate_motion_categories(
        flows,
        ground_truth,
        valid_mask,
        gt_magnitude
    )

    # Save numerical results
    save_results_csv(results)

    # Save plots
    print("\nGenerating visualizations...")

    plot_mean_epe(results)
    plot_fl_all(results)

    plot_epe_distribution(
        flows,
        ground_truth,
        valid_mask,
        gt_magnitude
    )

    save_motion_magnitude_map(
        gt_magnitude,
        valid_mask
    )

    # Save summary
    save_summary(
        results,
        runtimes,
        valid_mask,
        gt_magnitude
    )

    print("\n" + "=" * 75)
    print("TASK 4 COMPLETED SUCCESSFULLY")
    print("Results saved to:", OUTPUT_DIR)
    print("=" * 75)


if __name__ == "__main__":
    main()
