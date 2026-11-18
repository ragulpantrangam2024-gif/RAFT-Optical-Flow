
import cv2
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Patch


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

# KITTI frame pair: 000000_10 -> 000000_11
IMAGE_1_PATH = KITTI_ROOT / "image_2" / "000000_10.png"
IMAGE_2_PATH = KITTI_ROOT / "image_2" / "000000_11.png"

# Ground-truth optical flow
GROUND_TRUTH_PATH = (
    KITTI_ROOT / "flow_occ" / "000000_10.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "Stage_3" / "results" / "Task 2"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Horn-Schunck parameters
HS_ALPHA = 1.0
HS_ITERATIONS = 100


# ============================================================
# 2. LOAD KITTI IMAGE PAIR
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

    return image_1, gray_1, gray_2


# ============================================================
# 3. DECODE KITTI GROUND-TRUTH FLOW PNG
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

    # OpenCV loads the PNG in BGR order.
    #
    # Channel 2 = Red   = horizontal flow (u)
    # Channel 1 = Green = vertical flow (v)
    # Channel 0 = Blue  = validity information

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
# 4. FARNEBACK DENSE OPTICAL FLOW
# ============================================================

def calculate_farneback(gray_1, gray_2):

    flow = cv2.calcOpticalFlowFarneback(
        gray_1,
        gray_2,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    return flow


# ============================================================
# 5. HORN-SCHUNCK DENSE OPTICAL FLOW
# ============================================================

def calculate_horn_schunck(
    gray_1,
    gray_2,
    alpha=HS_ALPHA,
    iterations=HS_ITERATIONS
):

    image_1 = gray_1.astype(np.float32) / 255.0
    image_2 = gray_2.astype(np.float32) / 255.0

    average_image = (
        image_1 + image_2
    ) / 2.0

    # Calculate spatial gradients
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

    # Temporal gradient
    It = image_2 - image_1

    # Initialize flow
    u = np.zeros_like(image_1)
    v = np.zeros_like(image_1)

    # Neighbourhood averaging kernel
    kernel = np.array(
        [
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 6,  0,    1 / 6],
            [1 / 12, 1 / 6, 1 / 12]
        ],
        dtype=np.float32
    )

    alpha_squared = alpha ** 2

    # Iterative Horn-Schunck estimation
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
# 6. TV-L1 DENSE OPTICAL FLOW
# ============================================================

def calculate_tvl1(gray_1, gray_2):

    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            "OpenCV contrib module is unavailable. "
            "Install opencv-contrib-python in your environment."
        )

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

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

    flow = tvl1.calc(
        gray_1,
        gray_2,
        None
    )

    return flow


# ============================================================
# 7. DIS MEDIUM DENSE OPTICAL FLOW
# ============================================================

def calculate_dis(gray_1, gray_2):

    dis = cv2.DISOpticalFlow_create(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
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
# 8. CALCULATE KITTI EVALUATION METRICS
# ============================================================

def calculate_metrics(
    predicted_flow,
    ground_truth,
    valid_mask
):

    # Predicted flow components
    pred_u = predicted_flow[:, :, 0]
    pred_v = predicted_flow[:, :, 1]

    # Ground-truth flow components
    gt_u = ground_truth[:, :, 0]
    gt_v = ground_truth[:, :, 1]

    # --------------------------------------------------------
    # Endpoint Error
    # --------------------------------------------------------

    error_u = pred_u - gt_u
    error_v = pred_v - gt_v

    epe = np.sqrt(
        error_u**2 + error_v**2
    )

    # --------------------------------------------------------
    # Ground-truth motion magnitude
    # --------------------------------------------------------

    gt_magnitude = np.sqrt(
        gt_u**2 + gt_v**2
    )

    # --------------------------------------------------------
    # KITTI Fl-all outlier criterion
    #
    # EPE > 3 pixels AND
    # relative error > 5 percent
    # --------------------------------------------------------

    outlier_mask = (
        (epe > 3.0)
        &
        (epe > 0.05 * gt_magnitude)
    )

    # Evaluate only valid ground-truth pixels
    valid_epe = epe[valid_mask]

    valid_outliers = outlier_mask[valid_mask]

    if valid_epe.size == 0:
        raise ValueError(
            "No valid KITTI ground-truth pixels."
        )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = {

        "Valid Pixels": int(
            valid_epe.size
        ),

        "Mean EPE": float(
            np.mean(valid_epe)
        ),

        "Median EPE": float(
            np.median(valid_epe)
        ),

        "Standard Deviation": float(
            np.std(valid_epe)
        ),

        "Minimum EPE": float(
            np.min(valid_epe)
        ),

        "Maximum EPE": float(
            np.max(valid_epe)
        ),

        "95th Percentile EPE": float(
            np.percentile(valid_epe, 95)
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

    return metrics, epe, outlier_mask


# ============================================================
# 9. SAVE ENDPOINT ERROR VISUALIZATION
# ============================================================

def save_error_visualization(
    epe,
    valid_mask,
    method_name
):

    display_epe = np.where(
        valid_mask,
        epe,
        np.nan
    )

    plt.figure(figsize=(12, 5))

    image = plt.imshow(
        display_epe,
        cmap="turbo"
    )

    plt.colorbar(
        image,
        label="Endpoint Error (pixels)"
    )

    plt.title(
        f"{method_name} - KITTI Endpoint Error"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{method_name}_epe.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 10. SAVE IMPROVED FL-ALL OUTLIER VISUALIZATION
# ============================================================

def save_outlier_visualization(
    outlier_mask,
    valid_mask,
    method_name
):

    height, width = valid_mask.shape

    # Initialize RGB image
    visualization = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    # Invalid / unlabeled pixels: light gray
    visualization[~valid_mask] = [
        210, 210, 210
    ]

    # Valid and correctly estimated pixels: blue
    correct_mask = (
        valid_mask & (~outlier_mask)
    )

    visualization[correct_mask] = [
        40, 100, 180
    ]

    # Valid but incorrect pixels: red
    incorrect_mask = (
        valid_mask & outlier_mask
    )

    visualization[incorrect_mask] = [
        230, 50, 50
    ]

    plt.figure(figsize=(12, 5))

    plt.imshow(visualization)

    plt.title(
        f"{method_name} - KITTI Fl-all Outlier Map"
    )

    legend_items = [

        Patch(
            facecolor="#2864b4",
            label="Valid - Correct"
        ),

        Patch(
            facecolor="#e63232",
            label="Valid - Outlier"
        ),

        Patch(
            facecolor="#d2d2d2",
            label="Invalid / Unlabeled"
        )
    ]

    plt.legend(
        handles=legend_items,
        loc="lower right"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{method_name}_outliers.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. SAVE OPTICAL FLOW COLOR VISUALIZATION
# ============================================================

def save_flow_visualization(
    flow,
    method_name
):

    magnitude, angle = cv2.cartToPolar(
        flow[:, :, 0],
        flow[:, :, 1]
    )

    hsv = np.zeros(
        (
            flow.shape[0],
            flow.shape[1],
            3
        ),
        dtype=np.uint8
    )

    hsv[:, :, 0] = (
        angle * 180 / np.pi / 2
    ).astype(np.uint8)

    hsv[:, :, 1] = 255

    hsv[:, :, 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    flow_rgb = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )

    plt.figure(figsize=(12, 5))

    plt.imshow(flow_rgb)

    plt.title(
        f"{method_name} - KITTI Optical Flow"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{method_name}_flow.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. SAVE RESULTS TO CSV
# ============================================================

def save_csv_results(all_results):

    csv_path = (
        OUTPUT_DIR / "kitti_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\nCSV saved:", csv_path)


# ============================================================
# 13. SAVE SUMMARY REPORT
# ============================================================

def save_summary_report(
    all_results,
    valid_pixels
):

    summary_path = (
        OUTPUT_DIR / "kitti_evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STAGE 3 - TASK 2: KITTI OPTICAL FLOW EVALUATION\n"
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
            "Frame pair: 000000_10 -> 000000_11\n"
        )

        file.write(
            "Ground truth: flow_occ\n"
        )

        file.write(
            f"Valid pixels: {valid_pixels}\n"
        )

        file.write(
            "Evaluation: Mean EPE and KITTI Fl-all\n"
        )

        for result in all_results:

            file.write("\n\n")

            file.write(
                f"Algorithm: {result['Algorithm']}\n"
            )

            file.write("-" * 45 + "\n")

            for key, value in result.items():

                if key != "Algorithm":
                    file.write(
                        f"{key}: {value}\n"
                    )

    print("Summary saved:", summary_path)


# ============================================================
# 14. MAIN EXECUTION
# ============================================================

def main():

    print("\nLoading KITTI 2015 images...")

    image_1, gray_1, gray_2 = load_images()

    print(
        "Image shape:",
        image_1.shape
    )

    print("\nLoading KITTI ground-truth flow...")

    ground_truth, valid_mask = read_kitti_flow(
        GROUND_TRUTH_PATH
    )

    if ground_truth.shape[:2] != gray_1.shape:
        raise ValueError(
            "KITTI image and ground-truth dimensions do not match."
        )

    valid_pixels = int(
        np.count_nonzero(valid_mask)
    )

    print(
        "Ground-truth shape:",
        ground_truth.shape
    )

    print(
        "Valid pixels:",
        valid_pixels
    )

    # --------------------------------------------------------
    # Algorithms to evaluate
    # --------------------------------------------------------

    algorithms = {

        "Farneback": calculate_farneback,

        "Horn_Schunck": calculate_horn_schunck,

        "TVL1": calculate_tvl1,

        "DIS_Medium": calculate_dis
    }

    all_results = []

    print("\nStarting KITTI Optical Flow Evaluation")

    print("=" * 65)

    # --------------------------------------------------------
    # Evaluate each algorithm
    # --------------------------------------------------------

    for method_name, algorithm in algorithms.items():

        print(f"\nEvaluating: {method_name}")

        start_time = time.perf_counter()

        predicted_flow = algorithm(
            gray_1,
            gray_2
        )

        runtime = (
            time.perf_counter() - start_time
        )

        predicted_flow = predicted_flow.astype(
            np.float32
        )

        if predicted_flow.shape != ground_truth.shape:
            raise ValueError(
                f"Flow shape mismatch for {method_name}: "
                f"{predicted_flow.shape}"
            )

        # Calculate evaluation metrics
        metrics, epe, outlier_mask = calculate_metrics(
            predicted_flow,
            ground_truth,
            valid_mask
        )

        metrics["Runtime (seconds)"] = float(
            runtime
        )

        result = {
            "Algorithm": method_name,
            **metrics
        }

        all_results.append(result)

        # Save predicted flow array
        np.save(
            OUTPUT_DIR / f"{method_name}_flow.npy",
            predicted_flow
        )

        # Save flow visualization
        save_flow_visualization(
            predicted_flow,
            method_name
        )

        # Save EPE visualization
        save_error_visualization(
            epe,
            valid_mask,
            method_name
        )

        # Save improved outlier visualization
        save_outlier_visualization(
            outlier_mask,
            valid_mask,
            method_name
        )

        # Display metrics
        for key, value in metrics.items():

            if isinstance(value, float):
                print(
                    f"{key}: {value:.6f}"
                )

            else:
                print(
                    f"{key}: {value}"
                )

    # --------------------------------------------------------
    # Save CSV and summary
    # --------------------------------------------------------

    save_csv_results(all_results)

    save_summary_report(
        all_results,
        valid_pixels
    )

    print("\n" + "=" * 65)

    print(
        "KITTI TASK 2 COMPLETED SUCCESSFULLY"
    )

    print(
        "Results saved to:",
        OUTPUT_DIR
    )

    print("=" * 65)


# ============================================================
# 15. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
