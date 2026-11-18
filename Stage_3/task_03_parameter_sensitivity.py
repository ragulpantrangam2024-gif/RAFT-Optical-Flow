
import cv2
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
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

GROUND_TRUTH_PATH = (
    KITTI_ROOT / "flow_occ" / "000000_10.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Stage_3"
    / "results"
    / "Task 3"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. EXPERIMENT CONFIGURATIONS
# ============================================================

FARNEBACK_WINDOW_SIZES = [9, 15, 21]

HORN_SCHUNCK_ALPHAS = [0.5, 1.0, 2.0]

TVL1_LAMBDAS = [0.10, 0.15, 0.20]

DIS_PRESETS = {
    "ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "MEDIUM": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
}


# ============================================================
# 3. LOAD KITTI IMAGES
# ============================================================

def load_images():

    image_1 = cv2.imread(
        str(IMAGE_1_PATH),
        cv2.IMREAD_COLOR
    )

    image_2 = cv2.imread(
        str(IMAGE_2_PATH),
        cv2.IMREAD_COLOR
    )

    if image_1 is None or image_2 is None:
        raise FileNotFoundError(
            "Could not load KITTI image pair. "
            "Check the image_2 folder and filenames."
        )

    if image_1.shape != image_2.shape:
        raise ValueError(
            "KITTI image dimensions do not match."
        )

    gray_1 = cv2.cvtColor(
        image_1,
        cv2.COLOR_BGR2GRAY
    )

    gray_2 = cv2.cvtColor(
        image_2,
        cv2.COLOR_BGR2GRAY
    )

    return gray_1, gray_2


# ============================================================
# 4. LOAD AND DECODE KITTI GROUND-TRUTH FLOW
# ============================================================

def read_kitti_flow(file_path):

    encoded_flow = cv2.imread(
        str(file_path),
        cv2.IMREAD_UNCHANGED
    )

    if encoded_flow is None:
        raise FileNotFoundError(
            f"Ground-truth flow not found: {file_path}"
        )

    if encoded_flow.dtype != np.uint16:
        raise ValueError(
            "Expected a 16-bit KITTI flow PNG."
        )

    if (
        encoded_flow.ndim != 3
        or encoded_flow.shape[2] != 3
    ):
        raise ValueError(
            "Expected a three-channel KITTI flow PNG."
        )

    # OpenCV uses BGR channel ordering.
    # Red   = horizontal flow
    # Green = vertical flow
    # Blue  = validity

    flow_u = (
        encoded_flow[:, :, 2].astype(np.float32)
        - 32768.0
    ) / 64.0

    flow_v = (
        encoded_flow[:, :, 1].astype(np.float32)
        - 32768.0
    ) / 64.0

    valid_mask = encoded_flow[:, :, 0] > 0

    ground_truth = np.stack(
        (flow_u, flow_v),
        axis=-1
    )

    return ground_truth, valid_mask


# ============================================================
# 5. FARNEBACK OPTICAL FLOW
# ============================================================

def calculate_farneback(
    gray_1,
    gray_2,
    winsize
):

    flow = cv2.calcOpticalFlowFarneback(
        gray_1,
        gray_2,
        None,
        0.5,
        3,
        winsize,
        3,
        5,
        1.2,
        0
    )

    return flow


# ============================================================
# 6. HORN-SCHUNCK OPTICAL FLOW
# ============================================================

def calculate_horn_schunck(
    gray_1,
    gray_2,
    alpha,
    iterations=100
):

    image_1 = gray_1.astype(np.float32) / 255.0
    image_2 = gray_2.astype(np.float32) / 255.0

    average_image = (
        image_1 + image_2
    ) / 2.0

    Ix = cv2.Sobel(
        average_image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        scale=1 / 8
    )

    Iy = cv2.Sobel(
        average_image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        scale=1 / 8
    )

    It = image_2 - image_1

    u = np.zeros_like(image_1)
    v = np.zeros_like(image_1)

    kernel = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6,  0,    1 / 6],
            [1 / 12, 1 / 6, 1 / 12]
        ],
        dtype=np.float32
    )

    alpha_squared = alpha ** 2

    for _ in range(iterations):

        u_average = cv2.filter2D(
            u, -1, kernel
        )

        v_average = cv2.filter2D(
            v, -1, kernel
        )

        numerator = (
            Ix * u_average
            + Iy * v_average
            + It
        )

        denominator = (
            alpha_squared
            + Ix**2
            + Iy**2
        )

        correction = numerator / denominator

        u = u_average - Ix * correction
        v = v_average - Iy * correction

    return np.stack(
        (u, v),
        axis=-1
    )


# ============================================================
# 7. TV-L1 OPTICAL FLOW
# ============================================================

def calculate_tvl1(
    gray_1,
    gray_2,
    lambda_value
):

    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            "OpenCV contrib module is unavailable. "
            "Install opencv-contrib-python."
        )

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    tvl1.setTau(0.25)
    tvl1.setLambda(lambda_value)
    tvl1.setTheta(0.3)
    tvl1.setScalesNumber(5)
    tvl1.setWarpingsNumber(5)
    tvl1.setEpsilon(0.01)
    tvl1.setInnerIterations(30)
    tvl1.setOuterIterations(10)
    tvl1.setScaleStep(0.8)
    tvl1.setGamma(0.0)
    tvl1.setMedianFiltering(5)

    flow = tvl1.calc(
        gray_1,
        gray_2,
        None
    )

    return flow


# ============================================================
# 8. DIS OPTICAL FLOW
# ============================================================

def calculate_dis(
    gray_1,
    gray_2,
    preset
):

    dis = cv2.DISOpticalFlow_create(
        preset
    )

    dis.setUseSpatialPropagation(True)

    dis.setVariationalRefinementIterations(5)
    dis.setVariationalRefinementAlpha(20.0)
    dis.setVariationalRefinementDelta(5.0)
    dis.setVariationalRefinementGamma(10.0)

    flow = dis.calc(
        gray_1,
        gray_2,
        None
    )

    return flow


# ============================================================
# 9. EVALUATION METRICS
# ============================================================

def calculate_metrics(
    predicted_flow,
    ground_truth,
    valid_mask
):

    if predicted_flow.shape != ground_truth.shape:
        raise ValueError(
            "Predicted flow and ground truth dimensions differ."
        )

    error_u = (
        predicted_flow[:, :, 0]
        - ground_truth[:, :, 0]
    )

    error_v = (
        predicted_flow[:, :, 1]
        - ground_truth[:, :, 1]
    )

    epe = np.sqrt(
        error_u**2 + error_v**2
    )

    gt_magnitude = np.sqrt(
        ground_truth[:, :, 0]**2
        + ground_truth[:, :, 1]**2
    )

    # KITTI Fl-all:
    # EPE > 3 pixels AND relative error > 5%

    outlier_mask = (
        (epe > 3.0)
        &
        (epe > 0.05 * gt_magnitude)
    )

    valid_epe = epe[valid_mask]

    valid_outliers = outlier_mask[valid_mask]

    if valid_epe.size == 0:
        raise ValueError(
            "No valid ground-truth pixels."
        )

    metrics = {

        "Valid Pixels": int(valid_epe.size),

        "Mean EPE": float(
            np.mean(valid_epe)
        ),

        "Median EPE": float(
            np.median(valid_epe)
        ),

        "Fl-all (%)": float(
            np.mean(valid_outliers) * 100
        ),

        "EPE Below 1 Pixel (%)": float(
            np.mean(valid_epe < 1.0) * 100
        ),

        "EPE Below 3 Pixels (%)": float(
            np.mean(valid_epe < 3.0) * 100
        )
    }

    return metrics


# ============================================================
# 10. RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    algorithm_name,
    parameter_name,
    parameter_value,
    algorithm_function,
    gray_1,
    gray_2,
    ground_truth,
    valid_mask
):

    print(
        f"\n{algorithm_name} | "
        f"{parameter_name} = {parameter_value}"
    )

    start_time = time.perf_counter()

    predicted_flow = algorithm_function(
        gray_1,
        gray_2
    )

    runtime = (
        time.perf_counter() - start_time
    )

    predicted_flow = predicted_flow.astype(
        np.float32
    )

    metrics = calculate_metrics(
        predicted_flow,
        ground_truth,
        valid_mask
    )

    result = {

        "Algorithm": algorithm_name,

        "Parameter": parameter_name,

        "Parameter Value": parameter_value,

        **metrics,

        "Runtime (seconds)": float(runtime)
    }

    print(
        f"Mean EPE: {metrics['Mean EPE']:.6f}"
    )

    print(
        f"Fl-all: {metrics['Fl-all (%)']:.6f}%"
    )

    print(
        f"Runtime: {runtime:.6f} seconds"
    )

    return result


# ============================================================
# 11. RUN FARNEBACK SENSITIVITY EXPERIMENT
# ============================================================

def run_farneback_experiments(
    gray_1,
    gray_2,
    ground_truth,
    valid_mask
):

    results = []

    for winsize in FARNEBACK_WINDOW_SIZES:

        result = run_experiment(
            "Farneback",
            "winsize",
            winsize,
            lambda a, b: calculate_farneback(
                a, b, winsize
            ),
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )

        results.append(result)

    return results


# ============================================================
# 12. RUN HORN-SCHUNCK SENSITIVITY EXPERIMENT
# ============================================================

def run_horn_schunck_experiments(
    gray_1,
    gray_2,
    ground_truth,
    valid_mask
):

    results = []

    for alpha in HORN_SCHUNCK_ALPHAS:

        result = run_experiment(
            "Horn_Schunck",
            "alpha",
            alpha,
            lambda a, b: calculate_horn_schunck(
                a, b, alpha
            ),
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )

        results.append(result)

    return results


# ============================================================
# 13. RUN TV-L1 SENSITIVITY EXPERIMENT
# ============================================================

def run_tvl1_experiments(
    gray_1,
    gray_2,
    ground_truth,
    valid_mask
):

    results = []

    for lambda_value in TVL1_LAMBDAS:

        result = run_experiment(
            "TVL1",
            "lambda",
            lambda_value,
            lambda a, b: calculate_tvl1(
                a, b, lambda_value
            ),
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )

        results.append(result)

    return results


# ============================================================
# 14. RUN DIS PRESET EXPERIMENT
# ============================================================

def run_dis_experiments(
    gray_1,
    gray_2,
    ground_truth,
    valid_mask
):

    results = []

    for preset_name, preset_value in DIS_PRESETS.items():

        result = run_experiment(
            "DIS",
            "preset",
            preset_name,
            lambda a, b: calculate_dis(
                a, b, preset_value
            ),
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )

        results.append(result)

    return results


# ============================================================
# 15. SAVE RESULTS TO CSV
# ============================================================

def save_results_csv(results):

    csv_path = (
        OUTPUT_DIR / "parameter_sensitivity_metrics.csv"
    )

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
# 16. CREATE PARAMETER SENSITIVITY PLOTS
# ============================================================

def plot_numeric_sensitivity(
    results,
    algorithm_name,
    parameter_name,
    output_prefix
):

    algorithm_results = [
        result for result in results
        if result["Algorithm"] == algorithm_name
    ]

    x_values = np.array(
        [
            float(result["Parameter Value"])
            for result in algorithm_results
        ]
    )

    epe_values = np.array(
        [
            result["Mean EPE"]
            for result in algorithm_results
        ]
    )

    fl_values = np.array(
        [
            result["Fl-all (%)"]
            for result in algorithm_results
        ]
    )

    runtime_values = np.array(
        [
            result["Runtime (seconds)"]
            for result in algorithm_results
        ]
    )

    # EPE and Fl-all plot
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        x_values,
        epe_values,
        marker="o",
        linewidth=2,
        label="Mean EPE"
    )

    ax1.set_xlabel(parameter_name)
    ax1.set_ylabel("Mean EPE (pixels)")
    ax1.set_title(
        f"{algorithm_name} - {parameter_name} Sensitivity"
    )

    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        x_values,
        fl_values,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Fl-all"
    )

    ax2.set_ylabel("Fl-all (%)")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{output_prefix}_accuracy.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Runtime plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        x_values,
        runtime_values,
        marker="o",
        linewidth=2
    )

    plt.xlabel(parameter_name)
    plt.ylabel("Runtime (seconds)")

    plt.title(
        f"{algorithm_name} - Runtime Sensitivity"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{output_prefix}_runtime.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 17. CREATE DIS PRESET COMPARISON PLOTS
# ============================================================

def plot_dis_sensitivity(results):

    dis_results = [
        result for result in results
        if result["Algorithm"] == "DIS"
    ]

    names = [
        result["Parameter Value"]
        for result in dis_results
    ]

    epe_values = [
        result["Mean EPE"]
        for result in dis_results
    ]

    fl_values = [
        result["Fl-all (%)"]
        for result in dis_results
    ]

    runtime_values = [
        result["Runtime (seconds)"]
        for result in dis_results
    ]

    # Accuracy comparison
    fig, ax1 = plt.subplots(figsize=(9, 5))

    x = np.arange(len(names))

    ax1.bar(
        x - 0.18,
        epe_values,
        width=0.36,
        label="Mean EPE"
    )

    ax1.set_ylabel("Mean EPE (pixels)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)

    ax2 = ax1.twinx()

    ax2.bar(
        x + 0.18,
        fl_values,
        width=0.36,
        label="Fl-all"
    )

    ax2.set_ylabel("Fl-all (%)")

    ax1.set_title(
        "DIS - Preset Accuracy Comparison"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="best"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "dis_preset_accuracy.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # Runtime comparison
    plt.figure(figsize=(8, 5))

    plt.bar(
        names,
        runtime_values
    )

    plt.xlabel("DIS Preset")
    plt.ylabel("Runtime (seconds)")

    plt.title(
        "DIS - Preset Runtime Comparison"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "dis_preset_runtime.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 18. SAVE EXPERIMENT SUMMARY
# ============================================================

def save_summary(results):

    summary_path = (
        OUTPUT_DIR / "parameter_sensitivity_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STAGE 3 - TASK 3: PARAMETER SENSITIVITY ANALYSIS\n"
        )

        file.write("=" * 65 + "\n\n")

        file.write(
            "Dataset: KITTI 2015 Scene Flow\n"
        )

        file.write(
            "Split: Training\n"
        )

        file.write(
            "Sequence: 000000\n"
        )

        file.write(
            "Frame pair: 000000_10 -> 000000_11\n\n"
        )

        file.write(
            "Experiments: Farneback, Horn-Schunck, TV-L1 and DIS\n"
        )

        file.write(
            "Method: One parameter varied at a time\n"
        )

        for algorithm in [
            "Farneback",
            "Horn_Schunck",
            "TVL1",
            "DIS"
        ]:

            subset = [
                result for result in results
                if result["Algorithm"] == algorithm
            ]

            best = min(
                subset,
                key=lambda result: result["Mean EPE"]
            )

            file.write("\n\n")
            file.write(f"{algorithm}\n")
            file.write("-" * 45 + "\n")

            for result in subset:

                file.write(
                    f"{result['Parameter']} = "
                    f"{result['Parameter Value']}\n"
                )

                file.write(
                    f"Mean EPE: {result['Mean EPE']:.6f}\n"
                )

                file.write(
                    f"Fl-all: {result['Fl-all (%)']:.6f}%\n"
                )

                file.write(
                    f"Runtime: "
                    f"{result['Runtime (seconds)']:.6f} seconds\n\n"
                )

            file.write(
                "Best configuration by Mean EPE: "
                f"{best['Parameter']} = "
                f"{best['Parameter Value']}\n"
            )

            file.write(
                f"Best Mean EPE: {best['Mean EPE']:.6f}\n"
            )

            file.write(
                f"Corresponding Fl-all: "
                f"{best['Fl-all (%)']:.6f}%\n"
            )

    print("Summary saved:", summary_path)


# ============================================================
# 19. MAIN EXECUTION
# ============================================================

def main():

    print("\n" + "=" * 65)

    print(
        "STAGE 3 - TASK 3: PARAMETER SENSITIVITY ANALYSIS"
    )

    print("=" * 65)

    print("\nLoading KITTI images...")

    gray_1, gray_2 = load_images()

    print("Image shape:", gray_1.shape)

    print("\nLoading KITTI ground truth...")

    ground_truth, valid_mask = read_kitti_flow(
        GROUND_TRUTH_PATH
    )

    if ground_truth.shape[:2] != gray_1.shape:
        raise ValueError(
            "Image and ground-truth dimensions do not match."
        )

    print(
        "Ground-truth shape:",
        ground_truth.shape
    )

    print(
        "Valid pixels:",
        np.count_nonzero(valid_mask)
    )

    all_results = []

    # --------------------------------------------------------
    # Experiment 1: Farneback
    # --------------------------------------------------------

    print("\n\nEXPERIMENT 1: FARNEBACK")

    all_results.extend(
        run_farneback_experiments(
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )
    )

    # --------------------------------------------------------
    # Experiment 2: Horn-Schunck
    # --------------------------------------------------------

    print("\n\nEXPERIMENT 2: HORN-SCHUNCK")

    all_results.extend(
        run_horn_schunck_experiments(
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )
    )

    # --------------------------------------------------------
    # Experiment 3: TV-L1
    # --------------------------------------------------------

    print("\n\nEXPERIMENT 3: TV-L1")

    all_results.extend(
        run_tvl1_experiments(
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )
    )

    # --------------------------------------------------------
    # Experiment 4: DIS
    # --------------------------------------------------------

    print("\n\nEXPERIMENT 4: DIS")

    all_results.extend(
        run_dis_experiments(
            gray_1,
            gray_2,
            ground_truth,
            valid_mask
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    print("\nSaving experiment results...")

    save_results_csv(all_results)

    # --------------------------------------------------------
    # Generate plots
    # --------------------------------------------------------

    print("\nGenerating sensitivity plots...")

    plot_numeric_sensitivity(
        all_results,
        "Farneback",
        "Window size",
        "farneback"
    )

    plot_numeric_sensitivity(
        all_results,
        "Horn_Schunck",
        "Alpha",
        "horn_schunck"
    )

    plot_numeric_sensitivity(
        all_results,
        "TVL1",
        "Lambda",
        "tvl1"
    )

    plot_dis_sensitivity(all_results)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(all_results)

    print("\n" + "=" * 65)

    print(
        "TASK 3 COMPLETED SUCCESSFULLY"
    )

    print(
        "Total configurations evaluated:",
        len(all_results)
    )

    print(
        "Results saved to:",
        OUTPUT_DIR
    )

    print("=" * 65)


# ============================================================
# 20. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
