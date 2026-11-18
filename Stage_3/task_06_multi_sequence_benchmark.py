
"""
STAGE 3 - TASK 6: MULTI-SEQUENCE KITTI BENCHMARK

Algorithms:
    1. Farneback
    2. Horn-Schunck
    3. TV-L1
    4. DIS Medium

Dataset:
    KITTI 2015 Optical Flow - Training

Evaluation:
    - Mean Endpoint Error (EPE)
    - Median Endpoint Error
    - KITTI Fl-all (%)
    - Runtime

Outputs:
    Stage_3/results/Task 6/
"""

from pathlib import Path
import time
import csv

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

IMAGE_DIR = KITTI_ROOT / "image_2"
FLOW_DIR = KITTI_ROOT / "flow_occ"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Stage_3"
    / "results"
    / "Task 6"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ten evenly spaced KITTI training pairs.
# Increase this list later to evaluate more pairs.
PAIR_IDS = [
    "000000",
    "000020",
    "000040",
    "000060",
    "000080",
    "000100",
    "000120",
    "000140",
    "000160",
    "000180",
]

# Horn-Schunck parameters
HS_ALPHA = 1.0
HS_ITERATIONS = 100

# Farneback parameters
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


# ============================================================
# 2. LOAD KITTI IMAGES
# ============================================================

def load_kitti_images(pair_id):

    image1_path = IMAGE_DIR / f"{pair_id}_10.png"
    image2_path = IMAGE_DIR / f"{pair_id}_11.png"

    if not image1_path.exists():
        raise FileNotFoundError(f"Missing image: {image1_path}")

    if not image2_path.exists():
        raise FileNotFoundError(f"Missing image: {image2_path}")

    image1 = cv2.imread(
        str(image1_path),
        cv2.IMREAD_GRAYSCALE
    )

    image2 = cv2.imread(
        str(image2_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image1 is None or image2 is None:
        raise RuntimeError(
            f"Could not read images for pair {pair_id}"
        )

    if image1.shape != image2.shape:
        raise ValueError(
            f"Image dimensions differ for pair {pair_id}"
        )

    return image1, image2


# ============================================================
# 3. LOAD KITTI GROUND-TRUTH FLOW
# ============================================================

def load_kitti_ground_truth(pair_id):

    flow_path = FLOW_DIR / f"{pair_id}_10.png"

    if not flow_path.exists():
        raise FileNotFoundError(
            f"Missing ground-truth flow: {flow_path}"
        )

    flow_png = cv2.imread(
        str(flow_path),
        cv2.IMREAD_UNCHANGED
    )

    if flow_png is None:
        raise RuntimeError(
            f"Could not read ground truth: {flow_path}"
        )

    if flow_png.ndim != 3 or flow_png.shape[2] != 3:
        raise ValueError(
            f"Unexpected KITTI flow format: {flow_png.shape}"
        )

    # KITTI flow PNG encoding:
    # R = horizontal flow
    # G = vertical flow
    # B = validity

    flow_png = flow_png.astype(np.float32)

    ground_truth = np.zeros(
        (flow_png.shape[0], flow_png.shape[1], 2),
        dtype=np.float32
    )

    ground_truth[..., 0] = (
        flow_png[..., 2] - 32768.0
    ) / 64.0

    ground_truth[..., 1] = (
        flow_png[..., 1] - 32768.0
    ) / 64.0

    valid_mask = flow_png[..., 0] > 0

    return ground_truth, valid_mask


# ============================================================
# 4. FARNEBACK OPTICAL FLOW
# ============================================================

def calculate_farneback(image1, image2):

    return cv2.calcOpticalFlowFarneback(
        image1,
        image2,
        None,
        **FARNEBACK_PARAMS
    )


# ============================================================
# 5. HORN-SCHUNCK OPTICAL FLOW
# ============================================================

def calculate_horn_schunck(
    image1,
    image2,
    alpha=HS_ALPHA,
    iterations=HS_ITERATIONS
):

    image1_float = image1.astype(np.float32) / 255.0
    image2_float = image2.astype(np.float32) / 255.0

    # Spatial and temporal derivatives
    Ix = cv2.Sobel(
        image1_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    ) / 8.0

    Iy = cv2.Sobel(
        image1_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    ) / 8.0

    It = image2_float - image1_float

    u = np.zeros_like(image1_float)
    v = np.zeros_like(image1_float)

    # Horn-Schunck averaging kernel
    averaging_kernel = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6,  0.0,   1 / 6],
            [1 / 12, 1 / 6, 1 / 12],
        ],
        dtype=np.float32
    )

    alpha_squared = alpha ** 2

    for _ in range(iterations):

        u_average = cv2.filter2D(
            u,
            -1,
            averaging_kernel
        )

        v_average = cv2.filter2D(
            v,
            -1,
            averaging_kernel
        )

        derivative_term = (
            Ix * u_average
            + Iy * v_average
            + It
        )

        denominator = (
            alpha_squared
            + Ix ** 2
            + Iy ** 2
        )

        u = (
            u_average
            - Ix * derivative_term / denominator
        )

        v = (
            v_average
            - Iy * derivative_term / denominator
        )

    return np.dstack((u, v)).astype(np.float32)


# ============================================================
# 6. TV-L1 OPTICAL FLOW
# ============================================================

def calculate_tvl1(image1, image2):

    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            "OpenCV optflow module is unavailable. "
            "Install opencv-contrib-python."
        )

    if hasattr(
        cv2.optflow,
        "DualTVL1OpticalFlow_create"
    ):
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    elif hasattr(
        cv2.optflow,
        "createOptFlow_DualTVL1"
    ):
        tvl1 = cv2.optflow.createOptFlow_DualTVL1()

    else:
        raise RuntimeError(
            "Could not find the OpenCV TV-L1 constructor."
        )

    tvl1.setTau(0.25)
    tvl1.setLambda(0.15)
    tvl1.setTheta(0.3)
    tvl1.setScalesNumber(5)
    tvl1.setWarpingsNumber(5)
    tvl1.setEpsilon(0.01)
    tvl1.setInnerIterations(30)
    tvl1.setOuterIterations(10)
    tvl1.setScaleStep(0.8)
    tvl1.setGamma(0.0)
    tvl1.setMedianFiltering(5)

    return tvl1.calc(
        image1,
        image2,
        None
    )


# ============================================================
# 7. DIS MEDIUM OPTICAL FLOW
# ============================================================

def calculate_dis_medium(image1, image2):

    dis = cv2.DISOpticalFlow_create(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    )

    dis.setUseSpatialPropagation(True)

    dis.setVariationalRefinementIterations(5)
    dis.setVariationalRefinementAlpha(20.0)
    dis.setVariationalRefinementDelta(5.0)
    dis.setVariationalRefinementGamma(10.0)

    return dis.calc(
        image1,
        image2,
        None
    )


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

def calculate_metrics(
    predicted_flow,
    ground_truth,
    valid_mask
):

    if predicted_flow.shape != ground_truth.shape:
        raise ValueError(
            "Predicted and ground-truth flow shapes differ."
        )

    if valid_mask.shape != ground_truth.shape[:2]:
        raise ValueError(
            "Validity mask dimensions do not match flow."
        )

    valid_count = int(np.count_nonzero(valid_mask))

    if valid_count == 0:
        raise ValueError("No valid ground-truth pixels.")

    # Endpoint error
    difference = predicted_flow - ground_truth

    epe = np.linalg.norm(
        difference,
        axis=2
    )

    valid_epe = epe[valid_mask]

    # Ground-truth motion magnitude
    gt_magnitude = np.linalg.norm(
        ground_truth,
        axis=2
    )

    valid_gt_magnitude = gt_magnitude[valid_mask]

    # KITTI Fl-all:
    # EPE > 3 pixels AND relative error > 5%

    outlier_mask = (
        (epe > 3.0)
        &
        (
            epe
            / np.maximum(gt_magnitude, 1e-6)
            > 0.05
        )
    )

    valid_outliers = outlier_mask[valid_mask]

    metrics = {
        "valid_pixels": valid_count,
        "mean_epe": float(np.mean(valid_epe)),
        "median_epe": float(np.median(valid_epe)),
        "std_epe": float(np.std(valid_epe)),
        "max_epe": float(np.max(valid_epe)),
        "fl_all_percent": float(
            np.mean(valid_outliers) * 100.0
        ),
        "epe_below_1_percent": float(
            np.mean(valid_epe < 1.0) * 100.0
        ),
        "epe_below_3_percent": float(
            np.mean(valid_epe < 3.0) * 100.0
        ),
        "mean_gt_motion": float(
            np.mean(valid_gt_magnitude)
        ),
    }

    return metrics


# ============================================================
# 9. ALGORITHM REGISTRY
# ============================================================

ALGORITHMS = {
    "Farneback": calculate_farneback,
    "Horn-Schunck": calculate_horn_schunck,
    "TV-L1": calculate_tvl1,
    "DIS Medium": calculate_dis_medium,
}


# ============================================================
# 10. RUN ONE ALGORITHM AND MEASURE RUNTIME
# ============================================================

def evaluate_algorithm(
    algorithm_name,
    algorithm_function,
    image1,
    image2,
    ground_truth,
    valid_mask
):

    start_time = time.perf_counter()

    predicted_flow = algorithm_function(
        image1,
        image2
    )

    runtime = time.perf_counter() - start_time

    if predicted_flow is None:
        raise RuntimeError(
            f"{algorithm_name} returned no optical flow."
        )

    predicted_flow = np.asarray(
        predicted_flow,
        dtype=np.float32
    )

    if predicted_flow.shape != ground_truth.shape:
        raise ValueError(
            f"{algorithm_name} produced flow shape "
            f"{predicted_flow.shape}, expected "
            f"{ground_truth.shape}."
        )

    metrics = calculate_metrics(
        predicted_flow,
        ground_truth,
        valid_mask
    )

    metrics["runtime_seconds"] = float(runtime)

    return metrics


# ============================================================
# 11. BENCHMARK ALL PAIRS
# ============================================================

def run_benchmark():

    all_results = []

    print("=" * 75)
    print("STAGE 3 - TASK 6: MULTI-SEQUENCE KITTI BENCHMARK")
    print("=" * 75)

    print(f"\nKITTI directory: {KITTI_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of pairs: {len(PAIR_IDS)}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    if not FLOW_DIR.exists():
        raise FileNotFoundError(
            f"Ground-truth directory not found: {FLOW_DIR}"
        )

    for pair_number, pair_id in enumerate(PAIR_IDS, start=1):

        print("\n" + "=" * 75)
        print(
            f"PAIR {pair_number}/{len(PAIR_IDS)}: {pair_id}"
        )
        print("=" * 75)

        # Load images
        image1, image2 = load_kitti_images(pair_id)

        # Load ground truth
        ground_truth, valid_mask = (
            load_kitti_ground_truth(pair_id)
        )

        if image1.shape != ground_truth.shape[:2]:
            raise ValueError(
                f"Image and flow dimensions differ for {pair_id}"
            )

        print(f"Image dimensions: {image1.shape}")
        print(
            "Valid pixels:",
            np.count_nonzero(valid_mask)
        )

        # Evaluate every algorithm
        for algorithm_name, algorithm_function in (
            ALGORITHMS.items()
        ):

            print(f"\nRunning {algorithm_name}...")

            try:

                metrics = evaluate_algorithm(
                    algorithm_name,
                    algorithm_function,
                    image1,
                    image2,
                    ground_truth,
                    valid_mask
                )

                result = {
                    "pair_id": pair_id,
                    "algorithm": algorithm_name,
                    **metrics,
                }

                all_results.append(result)

                print(
                    f"Mean EPE: "
                    f"{metrics['mean_epe']:.6f}"
                )

                print(
                    f"Median EPE: "
                    f"{metrics['median_epe']:.6f}"
                )

                print(
                    f"Fl-all: "
                    f"{metrics['fl_all_percent']:.6f}%"
                )

                print(
                    f"Runtime: "
                    f"{metrics['runtime_seconds']:.4f} seconds"
                )

            except Exception as error:

                print(
                    f"ERROR: {algorithm_name} failed "
                    f"on pair {pair_id}: {error}"
                )

                raise

    return pd.DataFrame(all_results)


# ============================================================
# 12. GENERATE AGGREGATED SUMMARY
# ============================================================

def create_aggregate_summary(results_df):

    summary_df = (
        results_df
        .groupby("algorithm")
        .agg(
            evaluated_pairs=("pair_id", "nunique"),
            mean_epe=("mean_epe", "mean"),
            std_mean_epe=("mean_epe", "std"),
            mean_median_epe=("median_epe", "mean"),
            mean_fl_all=("fl_all_percent", "mean"),
            std_fl_all=("fl_all_percent", "std"),
            mean_runtime=("runtime_seconds", "mean"),
            std_runtime=("runtime_seconds", "std"),
            mean_valid_pixels=("valid_pixels", "mean"),
        )
        .reset_index()
    )

    return summary_df


# ============================================================
# 13. PLOT MEAN EPE
# ============================================================

def plot_mean_epe(summary_df):

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["algorithm"],
        summary_df["mean_epe"],
        yerr=summary_df["std_mean_epe"].fillna(0),
        capsize=5
    )

    plt.title(
        "Average Mean EPE Across KITTI Pairs"
    )

    plt.xlabel("Optical Flow Algorithm")
    plt.ylabel("Mean EPE (pixels)")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "average_mean_epe.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 14. PLOT FL-ALL
# ============================================================

def plot_fl_all(summary_df):

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["algorithm"],
        summary_df["mean_fl_all"],
        yerr=summary_df["std_fl_all"].fillna(0),
        capsize=5
    )

    plt.title(
        "Average KITTI Fl-all Across Pairs"
    )

    plt.xlabel("Optical Flow Algorithm")
    plt.ylabel("Fl-all (%)")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "average_fl_all.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 15. PLOT RUNTIME
# ============================================================

def plot_runtime(summary_df):

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary_df["algorithm"],
        summary_df["mean_runtime"],
        yerr=summary_df["std_runtime"].fillna(0),
        capsize=5
    )

    plt.title(
        "Average Runtime Across KITTI Pairs"
    )

    plt.xlabel("Optical Flow Algorithm")
    plt.ylabel("Runtime (seconds per pair)")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "average_runtime.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 16. PER-PAIR EPE COMPARISON
# ============================================================

def plot_pairwise_epe(results_df):

    pivot_df = results_df.pivot(
        index="pair_id",
        columns="algorithm",
        values="mean_epe"
    )

    ax = pivot_df.plot(
        kind="bar",
        figsize=(14, 7),
        width=0.8
    )

    ax.set_title(
        "Mean EPE Per KITTI Pair"
    )

    ax.set_xlabel("KITTI Pair ID")
    ax.set_ylabel("Mean EPE (pixels)")

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "mean_epe_per_pair.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 17. GENERATE TEXT SUMMARY
# ============================================================

def save_summary(
    results_df,
    summary_df
):

    summary_path = (
        OUTPUT_DIR
        / "multi_sequence_benchmark_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STAGE 3 - TASK 6: MULTI-SEQUENCE KITTI BENCHMARK\n"
        )

        file.write("=" * 65 + "\n\n")

        file.write(
            f"Number of evaluated pairs: "
            f"{results_df['pair_id'].nunique()}\n"
        )

        file.write(
            f"Algorithms: {', '.join(ALGORITHMS.keys())}\n\n"
        )

        file.write("Evaluated KITTI pairs:\n")

        for pair_id in PAIR_IDS:
            file.write(f"- {pair_id}\n")

        file.write("\nAGGREGATED RESULTS\n")
        file.write("=" * 65 + "\n\n")

        for _, row in summary_df.iterrows():

            file.write(
                f"Algorithm: {row['algorithm']}\n"
            )

            file.write(
                f"Evaluated pairs: "
                f"{int(row['evaluated_pairs'])}\n"
            )

            file.write(
                f"Average Mean EPE: "
                f"{row['mean_epe']:.6f} pixels\n"
            )

            file.write(
                f"Standard Deviation of Mean EPE: "
                f"{row['std_mean_epe']:.6f}\n"
            )

            file.write(
                f"Average Fl-all: "
                f"{row['mean_fl_all']:.6f}%\n"
            )

            file.write(
                f"Average Runtime: "
                f"{row['mean_runtime']:.6f} seconds\n\n"
            )

        best_epe = summary_df.loc[
            summary_df["mean_epe"].idxmin()
        ]

        best_fl = summary_df.loc[
            summary_df["mean_fl_all"].idxmin()
        ]

        fastest = summary_df.loc[
            summary_df["mean_runtime"].idxmin()
        ]

        file.write("BEST RESULTS\n")
        file.write("=" * 65 + "\n")

        file.write(
            f"Lowest average EPE: {best_epe['algorithm']}\n"
        )

        file.write(
            f"Lowest average Fl-all: {best_fl['algorithm']}\n"
        )

        file.write(
            f"Fastest average runtime: {fastest['algorithm']}\n"
        )

        file.write(
            "\nNote: Results are averages over the selected "
            "KITTI training pairs, not the complete KITTI benchmark.\n"
        )

        file.write(
            "Runtime is measured once per algorithm per pair "
            "and includes algorithm initialization where applicable.\n"
        )

    print(f"\nSummary saved: {summary_path}")


# ============================================================
# 18. MAIN
# ============================================================

def main():

    # Run benchmark
    results_df = run_benchmark()

    # Save per-pair results
    per_pair_csv = (
        OUTPUT_DIR
        / "multi_sequence_per_pair_metrics.csv"
    )

    results_df.to_csv(
        per_pair_csv,
        index=False,
        float_format="%.6f"
    )

    print(f"\nPer-pair CSV saved: {per_pair_csv}")

    # Aggregate results
    summary_df = create_aggregate_summary(
        results_df
    )

    aggregate_csv = (
        OUTPUT_DIR
        / "multi_sequence_aggregate_metrics.csv"
    )

    summary_df.to_csv(
        aggregate_csv,
        index=False,
        float_format="%.6f"
    )

    print(f"Aggregate CSV saved: {aggregate_csv}")

    # Generate plots
    print("\nGenerating benchmark visualizations...")

    plot_mean_epe(summary_df)
    plot_fl_all(summary_df)
    plot_runtime(summary_df)
    plot_pairwise_epe(results_df)

    # Save summary
    save_summary(
        results_df,
        summary_df
    )

    # Print final comparison
    print("\n" + "=" * 75)
    print("FINAL AGGREGATED BENCHMARK")
    print("=" * 75)

    print(
        summary_df[
            [
                "algorithm",
                "mean_epe",
                "mean_fl_all",
                "mean_runtime"
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 75)
    print("TASK 6 COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print(f"Evaluated pairs: {len(PAIR_IDS)}")
    print(f"Algorithms: {len(ALGORITHMS)}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
