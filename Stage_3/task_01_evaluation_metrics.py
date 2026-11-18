
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT / "training" / "flow" /
    "alley_1" / "frame_0001.flo"
)

INVALID_MASK_PATH = (
    PROJECT_ROOT / "training" / "invalid" /
    "alley_1" / "frame_0001.png"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "Stage_3" / "results" / "Task 1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DEFINE OPTICAL FLOW INPUTS
# ============================================================

FLOW_FILES = {
    "Farneback": (
        PROJECT_ROOT / "Stage_1" / "results" /
        "Task 4" / "farneback_flow.npy"
    ),

    "Horn_Schunck": (
        PROJECT_ROOT / "Stage_2" / "results" /
        "Task 1" / "horn_schunck_flow.npy"
    ),

    "TVL1": (
        PROJECT_ROOT / "Stage_2" / "results" /
        "Task 3" / "tvl1_flow.npy"
    ),

    "DIS_Medium": (
        PROJECT_ROOT / "Stage_2" / "results" /
        "Task 4" / "dis_medium_flow.npy"
    )
}


# ============================================================
# 3. READ SINTEL GROUND-TRUTH FLOW
# ============================================================

def read_flo_file(file_path):

    with open(file_path, "rb") as file:

        magic = np.fromfile(file, np.float32, count=1)[0]

        if magic != 202021.25:
            raise ValueError("Invalid Sintel .flo file.")

        width = np.fromfile(file, np.int32, count=1)[0]
        height = np.fromfile(file, np.int32, count=1)[0]

        flow = np.fromfile(
            file, np.float32, count=2 * width * height
        )

    if flow.size != 2 * width * height:
        raise ValueError("Incomplete ground-truth flow data.")

    return flow.reshape(height, width, 2)


# ============================================================
# 4. LOAD GROUND TRUTH AND VALID PIXEL MASK
# ============================================================

print("\nLoading Sintel ground truth...")

ground_truth = read_flo_file(GROUND_TRUTH_PATH)

invalid_mask = cv2.imread(
    str(INVALID_MASK_PATH),
    cv2.IMREAD_GRAYSCALE
)

if invalid_mask is None:
    raise FileNotFoundError(
        f"Invalid mask not found: {INVALID_MASK_PATH}"
    )

if invalid_mask.shape != ground_truth.shape[:2]:
    raise ValueError("Ground truth and invalid mask dimensions differ.")

# Zero-valued pixels are valid in this project's mask convention.
valid_mask = invalid_mask == 0

print("Ground truth shape:", ground_truth.shape)
print("Valid pixels:", np.count_nonzero(valid_mask))


# ============================================================
# 5. CALCULATE OPTICAL FLOW EVALUATION METRICS
# ============================================================

def calculate_metrics(predicted_flow, ground_truth, valid_mask):

    # Separate horizontal and vertical flow components
    pred_u = predicted_flow[:, :, 0]
    pred_v = predicted_flow[:, :, 1]

    gt_u = ground_truth[:, :, 0]
    gt_v = ground_truth[:, :, 1]

    # --------------------------------------------------------
    # Endpoint Error (EPE)
    # --------------------------------------------------------

    error_u = pred_u - gt_u
    error_v = pred_v - gt_v

    epe = np.sqrt(error_u**2 + error_v**2)

    # --------------------------------------------------------
    # Angular Error (Sintel convention)
    # --------------------------------------------------------

    numerator = (
        1.0 + pred_u * gt_u + pred_v * gt_v
    )

    denominator = (
        np.sqrt(1.0 + pred_u**2 + pred_v**2)
        * np.sqrt(1.0 + gt_u**2 + gt_v**2)
    )

    cosine = numerator / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    angular_error = np.degrees(
        np.arccos(cosine)
    )

    # --------------------------------------------------------
    # Select valid pixels
    # --------------------------------------------------------

    valid_epe = epe[valid_mask]
    valid_angular = angular_error[valid_mask]

    if valid_epe.size == 0:
        raise ValueError("No valid pixels found.")

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    metrics = {

        "Valid Pixels": int(valid_epe.size),

        "Mean EPE": float(np.mean(valid_epe)),

        "Median EPE": float(np.median(valid_epe)),

        "Standard Deviation": float(np.std(valid_epe)),

        "Minimum EPE": float(np.min(valid_epe)),

        "Maximum EPE": float(np.max(valid_epe)),

        "95th Percentile EPE": float(
            np.percentile(valid_epe, 95)
        ),

        "Mean Angular Error": float(
            np.mean(valid_angular)
        ),

        "EPE Below 1 Pixel (%)": float(
            np.mean(valid_epe < 1.0) * 100
        ),

        "EPE Below 3 Pixels (%)": float(
            np.mean(valid_epe < 3.0) * 100
        ),

        "EPE Below 5 Pixels (%)": float(
            np.mean(valid_epe < 5.0) * 100
        )
    }

    return metrics, epe


# ============================================================
# 6. GENERATE EPE VISUALIZATION
# ============================================================

def save_epe_visualization(
    epe, valid_mask, method_name
):

    display_epe = np.where(
        valid_mask, epe, np.nan
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
        f"{method_name} - Endpoint Error"
    )

    plt.axis("off")
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / f"{method_name}_epe.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. EVALUATE ALL OPTICAL FLOW ALGORITHMS
# ============================================================

def main():

    all_results = []

    print("\nStarting Optical Flow Evaluation")
    print("=" * 60)

    for method_name, flow_path in FLOW_FILES.items():

        print(f"\nEvaluating: {method_name}")

        if not flow_path.exists():
            raise FileNotFoundError(
                f"Flow file not found: {flow_path}"
            )

        # Load predicted flow
        predicted_flow = np.load(flow_path).astype(
            np.float32
        )

        # Verify dimensions
        if predicted_flow.shape != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch for {method_name}: "
                f"{predicted_flow.shape} vs "
                f"{ground_truth.shape}"
            )

        # Calculate metrics
        metrics, epe = calculate_metrics(
            predicted_flow,
            ground_truth,
            valid_mask
        )

        result = {
            "Algorithm": method_name,
            **metrics
        }

        all_results.append(result)

        # Generate EPE visualization
        save_epe_visualization(
            epe,
            valid_mask,
            method_name
        )

        # Display results
        for metric_name, value in metrics.items():

            if isinstance(value, float):
                print(f"{metric_name}: {value:.6f}")
            else:
                print(f"{metric_name}: {value}")

    # ========================================================
    # 8. SAVE RESULTS TO CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR / "evaluation_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # 9. SAVE SUMMARY REPORT
    # ========================================================

    summary_path = (
        OUTPUT_DIR / "evaluation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STAGE 3 - TASK 1: OPTICAL FLOW EVALUATION\n"
        )

        file.write("=" * 60 + "\n\n")

        file.write(
            f"Ground Truth: {GROUND_TRUTH_PATH}\n"
        )

        file.write(
            f"Valid Pixels: {np.count_nonzero(valid_mask)}\n"
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

    print("\n" + "=" * 60)
    print("TASK 1 COMPLETED SUCCESSFULLY")
    print("Results saved to:", OUTPUT_DIR)
    print("=" * 60)


# ============================================================
# 10. EXECUTION
# ============================================================

if __name__ == "__main__":
    main()
