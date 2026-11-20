
from pathlib import Path
import sys
import cv2
import csv
import struct
import inspect
import contextlib
import io

import numpy as np
import torch


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_5_DIR = PROJECT_ROOT / "Stage_5"

SINTEL_IMAGE_ROOT = (
    PROJECT_ROOT / "MPI-Sintel-training_images" / "training" / "final"
)
SINTEL_FLOW_ROOT = PROJECT_ROOT / "training" / "flow"
SINTEL_INVALID_ROOT = PROJECT_ROOT / "training" / "invalid"

KITTI_ROOT = (
    PROJECT_ROOT / "KITTI_Optical_Flow_2025"
    / "data_scene_flow" / "training"
)

CHECKPOINT_PATH = (
    STAGE_5_DIR / "results" / "Task 4" / "raft_best_model.pth"
)

OUTPUT_ROOT = STAGE_5_DIR / "results" / "Task 6"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_WIDTH = 400
MAX_SAMPLES_PER_DATASET = 10
NUM_ITERATIONS = 5

sys.path.insert(0, str(STAGE_5_DIR))
import task_02_raft_forward_pass as task2

RAFTForwardModel = task2.RAFTForwardModel


# ============================================================
# DATASET FILE DISCOVERY
# ============================================================

def discover_sintel_pairs():
    pairs = []

    for flow_path in sorted(SINTEL_FLOW_ROOT.glob("*/*.flo")):
        scene = flow_path.parent.name
        frame_number = int(flow_path.stem.split("_")[-1])

        image1 = SINTEL_IMAGE_ROOT / scene / f"frame_{frame_number:04d}.png"
        image2 = SINTEL_IMAGE_ROOT / scene / f"frame_{frame_number + 1:04d}.png"
        invalid_path = SINTEL_INVALID_ROOT / scene / f"frame_{frame_number:04d}.png"

        if image1.exists() and image2.exists() and invalid_path.exists():
            pairs.append({
                "image1": image1,
                "image2": image2,
                "flow": flow_path,
                "invalid": invalid_path,
                "name": f"{scene}_frame_{frame_number:04d}",
            })

    return pairs


def discover_kitti_pairs():
    pairs = []

    image_root = KITTI_ROOT / "image_2"
    flow_root = KITTI_ROOT / "flow_occ"

    for image1 in sorted(image_root.glob("*_10.png")):
        image2 = image_root / image1.name.replace("_10.png", "_11.png")
        flow_path = flow_root / image1.name

        if image2.exists() and flow_path.exists():
            pairs.append({
                "image1": image1,
                "image2": image2,
                "flow": flow_path,
                "name": image1.stem,
            })

    return pairs


def select_evenly_spaced(items, maximum):
    if len(items) <= maximum:
        return items

    indices = np.linspace(
        0, len(items) - 1, maximum, dtype=int
    )

    return [items[i] for i in indices]


# ============================================================
# GROUND-TRUTH FLOW LOADING
# ============================================================

def read_flo(path):
    with open(path, "rb") as file:
        magic = struct.unpack("f", file.read(4))[0]

        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file: {path}")

        width = struct.unpack("i", file.read(4))[0]
        height = struct.unpack("i", file.read(4))[0]

        data = np.fromfile(file, np.float32)

    return data.reshape(height, width, 2)


def load_sintel_ground_truth(pair):
    flow = read_flo(pair["flow"])

    invalid = cv2.imread(
        str(pair["invalid"]), cv2.IMREAD_GRAYSCALE
    )

    if invalid is None:
        raise FileNotFoundError(pair["invalid"])

    valid = invalid == 0
    valid &= np.isfinite(flow).all(axis=2)

    return flow.astype(np.float32), valid


def load_kitti_ground_truth(pair):
    # OpenCV reads the encoded KITTI PNG in BGR channel order.
    encoded = cv2.imread(
        str(pair["flow"]), cv2.IMREAD_UNCHANGED
    )

    if encoded is None:
        raise FileNotFoundError(pair["flow"])

    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(
            f"Unexpected KITTI flow image shape: {encoded.shape}"
        )

    encoded = cv2.cvtColor(encoded, cv2.COLOR_BGR2RGB)

    flow = np.zeros(
        (encoded.shape[0], encoded.shape[1], 2),
        dtype=np.float32,
    )

    flow[..., 0] = (encoded[..., 0].astype(np.float32) - 32768.0) / 64.0
    flow[..., 1] = (encoded[..., 1].astype(np.float32) - 32768.0) / 64.0

    valid = encoded[..., 2] > 0

    return flow, valid


# ============================================================
# MODEL AND IMAGE PREPROCESSING
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    model = RAFTForwardModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def preprocess_images(image1_path, image2_path):
    image1 = cv2.imread(str(image1_path))
    image2 = cv2.imread(str(image2_path))

    if image1 is None or image2 is None:
        raise FileNotFoundError(
            f"Could not read image pair: {image1_path}"
        )

    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Input image dimensions do not match.")

    original_height, original_width = image1.shape[:2]

    resized_height = int(round(
        (original_height * TARGET_WIDTH / original_width) / 8
    )) * 8
    resized_height = max(resized_height, 8)

    image1_resized = cv2.resize(
        image1, (TARGET_WIDTH, resized_height)
    )
    image2_resized = cv2.resize(
        image2, (TARGET_WIDTH, resized_height)
    )

    rgb1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB)

    tensor1 = torch.from_numpy(rgb1).float()
    tensor2 = torch.from_numpy(rgb2).float()

    tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0) / 255.0

    return (
        tensor1.to(DEVICE),
        tensor2.to(DEVICE),
        image1,
        image2,
    )


def extract_flow(output):
    if isinstance(output, dict):
        for key in (
            "final_flow", "flow", "flow_predictions", "predictions"
        ):
            if key in output:
                return extract_flow(output[key])

        raise ValueError(f"Unknown model output keys: {list(output.keys())}")

    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Empty model output.")
        return extract_flow(output[-1])

    if not torch.is_tensor(output):
        raise TypeError(f"Unexpected model output: {type(output)}")

    if output.ndim != 4:
        raise ValueError(f"Unexpected flow shape: {tuple(output.shape)}")

    if output.shape[1] == 2:
        flow = output[0].permute(1, 2, 0)
    elif output.shape[-1] == 2:
        flow = output[0]
    else:
        raise ValueError(f"Cannot identify flow channels: {output.shape}")

    return flow.detach().cpu().numpy()


def predict_flow(model, tensor1, tensor2, original_width, original_height):
    parameters = inspect.signature(model.forward).parameters
    kwargs = {}

    if "iterations" in parameters:
        kwargs["iterations"] = NUM_ITERATIONS
    elif "num_iterations" in parameters:
        kwargs["num_iterations"] = NUM_ITERATIONS
    elif "iters" in parameters:
        kwargs["iters"] = NUM_ITERATIONS

    with torch.no_grad():
        with contextlib.redirect_stdout(io.StringIO()):
            output = model(tensor1, tensor2, **kwargs)

    flow = extract_flow(output)

    # Resize flow to original resolution and scale displacement values.
    flow = cv2.resize(
        flow,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    flow[..., 0] *= original_width / tensor1.shape[-1]
    flow[..., 1] *= original_height / tensor1.shape[-2]

    return flow.astype(np.float32)


# ============================================================
# EVALUATION METRICS
# ============================================================

def calculate_metrics(predicted, ground_truth, valid):
    difference = predicted - ground_truth

    epe = np.sqrt(
        difference[..., 0] ** 2 + difference[..., 1] ** 2
    )

    valid_epe = epe[valid]

    if valid_epe.size == 0:
        raise ValueError("No valid ground-truth pixels found.")

    gt_magnitude = np.sqrt(
        ground_truth[..., 0] ** 2 + ground_truth[..., 1] ** 2
    )

    # KITTI Fl-all definition: EPE > 3 px AND relative error > 5%.
    relative_error = epe / np.maximum(gt_magnitude, 1e-6)
    fl_all = (
        (epe > 3.0) & (relative_error > 0.05) & valid
    ).sum() / valid.sum() * 100.0

    return {
        "valid_pixels": int(valid.sum()),
        "mean_epe": float(valid_epe.mean()),
        "median_epe": float(np.median(valid_epe)),
        "epe_below_1_percent": float((valid_epe < 1).mean() * 100),
        "epe_below_3_percent": float((valid_epe < 3).mean() * 100),
        "epe_below_5_percent": float((valid_epe < 5).mean() * 100),
        "fl_all_percent": float(fl_all),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def flow_to_color(flow):
    magnitude, angle = cv2.cartToPolar(
        flow[..., 0], flow[..., 1], angleInDegrees=True
    )

    hsv = np.zeros(
        (flow.shape[0], flow.shape[1], 3),
        dtype=np.uint8,
    )

    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255

    max_magnitude = np.percentile(magnitude, 99)

    if max_magnitude > 0:
        hsv[..., 2] = np.clip(
            magnitude / max_magnitude * 255, 0, 255
        ).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_visualizations(
    output_directory,
    image1,
    predicted,
    ground_truth,
    valid,
):
    output_directory.mkdir(parents=True, exist_ok=True)

    pred_color = flow_to_color(predicted)
    gt_color = flow_to_color(ground_truth)

    error = np.sqrt(
        (predicted[..., 0] - ground_truth[..., 0]) ** 2
        + (predicted[..., 1] - ground_truth[..., 1]) ** 2
    )

    error_display = np.zeros(error.shape, dtype=np.uint8)

    if valid.any():
        error_scale = np.percentile(error[valid], 99)
        if error_scale > 0:
            error_display[valid] = np.clip(
                error[valid] / error_scale * 255, 0, 255
            ).astype(np.uint8)

    error_color = cv2.applyColorMap(
        error_display, cv2.COLORMAP_TURBO
    )

    cv2.imwrite(
        str(output_directory / "input_frame_1.png"), image1
    )
    cv2.imwrite(
        str(output_directory / "predicted_flow.png"), pred_color
    )
    cv2.imwrite(
        str(output_directory / "ground_truth_flow.png"), gt_color
    )
    cv2.imwrite(
        str(output_directory / "endpoint_error_map.png"), error_color
    )

    np.save(
        output_directory / "predicted_flow.npy", predicted
    )


# ============================================================
# DATASET EVALUATION
# ============================================================

def evaluate_dataset(model, dataset_name, pairs, loader):
    print(f"\nEvaluating {dataset_name}: {len(pairs)} pairs")

    output_directory = OUTPUT_ROOT / dataset_name
    output_directory.mkdir(parents=True, exist_ok=True)

    records = []

    for index, pair in enumerate(pairs, start=1):
        tensor1, tensor2, image1, image2 = preprocess_images(
            pair["image1"], pair["image2"]
        )

        height, width = image1.shape[:2]

        predicted = predict_flow(
            model, tensor1, tensor2, width, height
        )

        ground_truth, valid = loader(pair)

        if predicted.shape[:2] != ground_truth.shape[:2]:
            raise ValueError(
                f"Prediction/GT size mismatch for {pair['name']}: "
                f"{predicted.shape} vs {ground_truth.shape}"
            )

        metrics = calculate_metrics(
            predicted, ground_truth, valid
        )

        record = {
            "sample": pair["name"],
            **metrics,
        }
        records.append(record)

        print(
            f"[{index:02d}/{len(pairs):02d}] {pair['name']} | "
            f"EPE: {metrics['mean_epe']:.4f} | "
            f"Fl-all: {metrics['fl_all_percent']:.2f}%"
        )

        # Save detailed visualization for the first evaluated sample.
        if index == 1:
            save_visualizations(
                output_directory / "visualization",
                image1,
                predicted,
                ground_truth,
                valid,
            )

    csv_path = output_directory / "per_sample_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(records)

    # Aggregate across valid pixels, rather than averaging sample means.
    total_valid = sum(r["valid_pixels"] for r in records)

    aggregate = {
        "dataset": dataset_name,
        "samples_evaluated": len(records),
        "total_valid_pixels": total_valid,
        "mean_epe_pixel_weighted": sum(
            r["mean_epe"] * r["valid_pixels"] for r in records
        ) / total_valid,
        "median_epe_mean_of_samples": float(
            np.mean([r["median_epe"] for r in records])
        ),
        "epe_below_1_mean_percent": float(
            np.mean([r["epe_below_1_percent"] for r in records])
        ),
        "epe_below_3_mean_percent": float(
            np.mean([r["epe_below_3_percent"] for r in records])
        ),
        "epe_below_5_mean_percent": float(
            np.mean([r["epe_below_5_percent"] for r in records])
        ),
        "fl_all_mean_percent": float(
            np.mean([r["fl_all_percent"] for r in records])
        ),
    }

    summary_path = output_directory / "evaluation_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        for key, value in aggregate.items():
            file.write(f"{key}: {value}\n")

    print(f"\n{dataset_name} aggregate results:")
    for key, value in aggregate.items():
        print(f"{key}: {value}")

    return aggregate


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("STAGE 5 - TASK 6: RAFT QUANTITATIVE EVALUATION")
    print("=" * 65)

    model = load_model()

    sintel_pairs = discover_sintel_pairs()
    kitti_pairs = discover_kitti_pairs()

    print(f"\nSintel pairs discovered: {len(sintel_pairs)}")
    print(f"KITTI pairs discovered: {len(kitti_pairs)}")

    if not sintel_pairs:
        raise RuntimeError("No valid Sintel image/flow pairs were found.")

    if not kitti_pairs:
        raise RuntimeError("No valid KITTI image/flow pairs were found.")

    sintel_selected = select_evenly_spaced(
        sintel_pairs, MAX_SAMPLES_PER_DATASET
    )
    kitti_selected = select_evenly_spaced(
        kitti_pairs, MAX_SAMPLES_PER_DATASET
    )

    print(f"Sintel selected: {len(sintel_selected)}")
    print(f"KITTI selected: {len(kitti_selected)}")

    sintel_result = evaluate_dataset(
        model,
        "Sintel",
        sintel_selected,
        load_sintel_ground_truth,
    )

    kitti_result = evaluate_dataset(
        model,
        "KITTI",
        kitti_selected,
        load_kitti_ground_truth,
    )

    combined_summary = OUTPUT_ROOT / "stage_5_task_6_summary.txt"

    with open(combined_summary, "w", encoding="utf-8") as file:
        file.write("STAGE 5 TASK 6 - QUANTITATIVE EVALUATION\n\n")

        for result in (sintel_result, kitti_result):
            file.write(f"{result['dataset']}\n")
            for key, value in result.items():
                if key != "dataset":
                    file.write(f"{key}: {value}\n")
            file.write("\n")

        file.write(
            "Note: Fl-all is computed using the KITTI outlier definition. "
            "It is included for both datasets as a common diagnostic, "
            "but it is the official KITTI metric only on KITTI data.\n"
        )

        file.write(
            "This evaluation uses an evenly spaced subset of available "
            "pairs. It is not guaranteed to reproduce the held-out "
            "validation split used during training.\n"
        )

    print("\n" + "=" * 65)
    print("TASK 6 COMPLETED")
    print(f"Results saved to: {OUTPUT_ROOT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
