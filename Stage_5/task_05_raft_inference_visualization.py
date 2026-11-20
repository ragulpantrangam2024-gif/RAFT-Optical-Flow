
from pathlib import Path
import sys
import inspect
import contextlib
import io

import cv2
import numpy as np
import torch


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_5_DIR = PROJECT_ROOT / "Stage_5"

TASK_2_SCRIPT = STAGE_5_DIR / "task_02_raft_forward_pass.py"

CHECKPOINT_PATH = (
    STAGE_5_DIR
    / "results"
    / "Task 4"
    / "raft_best_model.pth"
)

SINTEL_IMAGE_ROOT = (
    PROJECT_ROOT
    / "MPI-Sintel-training_images"
    / "training"
    / "final"
)

KITTI_IMAGE_ROOT = (
    PROJECT_ROOT
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
    / "image_2"
)

OUTPUT_ROOT = STAGE_5_DIR / "results" / "Task 5"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_WIDTH = 400
NUM_ITERATIONS = 5


# ============================================================
# IMPORT THE EXISTING RAFT MODEL
# ============================================================

sys.path.insert(0, str(STAGE_5_DIR))

if not TASK_2_SCRIPT.exists():
    raise FileNotFoundError(
        f"Task 2 model script not found: {TASK_2_SCRIPT}"
    )

import task_02_raft_forward_pass as task2


if not hasattr(task2, "RAFTForwardModel"):
    raise AttributeError(
        "RAFTForwardModel was not found in task_02_raft_forward_pass.py. "
        "Check the model class name in your Task 2 script."
    )

RAFTForwardModel = task2.RAFTForwardModel


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image_pair(image1_path, image2_path):
    image1 = cv2.imread(str(image1_path))
    image2 = cv2.imread(str(image2_path))

    if image1 is None:
        raise FileNotFoundError(f"Could not read: {image1_path}")

    if image2 is None:
        raise FileNotFoundError(f"Could not read: {image2_path}")

    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("The two input images must have the same dimensions.")

    original_height, original_width = image1.shape[:2]

    # Preserve aspect ratio and make height divisible by 8.
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

    image1_rgb = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB)

    tensor1 = torch.from_numpy(image1_rgb).float()
    tensor2 = torch.from_numpy(image2_rgb).float()

    tensor1 = tensor1.permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor2 = tensor2.permute(2, 0, 1).unsqueeze(0) / 255.0

    return (
        tensor1.to(DEVICE),
        tensor2.to(DEVICE),
        image1,
        image2,
        (original_width, original_height),
    )


# ============================================================
# MODEL OUTPUT HANDLING
# ============================================================

def extract_flow(output):
    """Extract the final flow prediction from common model output formats."""

    if isinstance(output, dict):
        for key in (
            "final_flow",
            "flow",
            "flow_predictions",
            "predictions",
        ):
            if key in output:
                return extract_flow(output[key])

        raise ValueError(
            f"Could not identify flow in model output keys: {list(output.keys())}"
        )

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise ValueError("The model returned an empty output.")

        return extract_flow(output[-1])

    if not torch.is_tensor(output):
        raise TypeError(
            f"Unsupported model output type: {type(output)}"
        )

    # Expected: [B, 2, H, W] or [B, H, W, 2].
    if output.ndim != 4:
        raise ValueError(
            f"Expected a 4D flow tensor, received shape {tuple(output.shape)}"
        )

    if output.shape[1] == 2:
        flow = output[0].permute(1, 2, 0)

    elif output.shape[-1] == 2:
        flow = output[0]

    else:
        raise ValueError(
            f"Could not identify the two flow channels: {tuple(output.shape)}"
        )

    return flow.detach().cpu().numpy()


def run_inference(model, image1_tensor, image2_tensor):
    """Run the existing model and return its final flow prediction."""

    forward_parameters = inspect.signature(model.forward).parameters

    kwargs = {}

    if "iterations" in forward_parameters:
        kwargs["iterations"] = NUM_ITERATIONS
    elif "num_iterations" in forward_parameters:
        kwargs["num_iterations"] = NUM_ITERATIONS
    elif "iters" in forward_parameters:
        kwargs["iters"] = NUM_ITERATIONS

    with torch.no_grad():
        # Suppress the iteration messages printed by the Task 2 model.
        with contextlib.redirect_stdout(io.StringIO()):
            output = model(image1_tensor, image2_tensor, **kwargs)

    return extract_flow(output)


# ============================================================
# FLOW RESIZING AND VISUALIZATION
# ============================================================

def resize_flow_to_original(flow, original_size):
    original_width, original_height = original_size

    flow_resized = cv2.resize(
        flow,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )

    scale_x = original_width / flow.shape[1]
    scale_y = original_height / flow.shape[0]

    flow_resized[..., 0] *= scale_x
    flow_resized[..., 1] *= scale_y

    return flow_resized


def flow_to_color(flow):
    """Convert a 2-channel optical flow field into an HSV visualization."""

    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=True)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255

    max_magnitude = np.percentile(magnitude, 99)

    if max_magnitude > 0:
        value = np.clip(
            magnitude / max_magnitude * 255,
            0,
            255,
        )
        hsv[..., 2] = value.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_results(
    dataset_name,
    flow,
    image1,
    image2,
    output_directory,
):
    output_directory.mkdir(parents=True, exist_ok=True)

    magnitude = np.sqrt(
        flow[..., 0] ** 2 + flow[..., 1] ** 2
    )

    flow_color = flow_to_color(flow)

    magnitude_image = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype(np.uint8)

    magnitude_image = cv2.applyColorMap(
        magnitude_image,
        cv2.COLORMAP_TURBO,
    )

    cv2.imwrite(
        str(output_directory / "input_frame_1.png"),
        image1,
    )

    cv2.imwrite(
        str(output_directory / "input_frame_2.png"),
        image2,
    )

    cv2.imwrite(
        str(output_directory / "predicted_flow_color.png"),
        flow_color,
    )

    cv2.imwrite(
        str(output_directory / "flow_magnitude.png"),
        magnitude_image,
    )

    np.save(
        output_directory / "predicted_flow.npy",
        flow.astype(np.float32),
    )

    summary = [
        f"Dataset: {dataset_name}",
        f"Flow shape: {flow.shape}",
        f"Mean flow magnitude: {magnitude.mean():.6f}",
        f"Median flow magnitude: {np.median(magnitude):.6f}",
        f"Maximum flow magnitude: {magnitude.max():.6f}",
        f"Mean horizontal flow: {flow[..., 0].mean():.6f}",
        f"Mean vertical flow: {flow[..., 1].mean():.6f}",
    ]

    summary_path = output_directory / "inference_summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print(f"\n{dataset_name} inference completed.")
    for line in summary:
        print(line)


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_trained_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = RAFTForwardModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unexpected checkpoint format.")

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


# ============================================================
# MAIN
# ============================================================

def process_pair(model, dataset_name, image1_path, image2_path):
    print(f"\nProcessing {dataset_name}...")

    (
        tensor1,
        tensor2,
        original_image1,
        original_image2,
        original_size,
    ) = load_image_pair(image1_path, image2_path)

    print(f"Original image size: {original_size}")
    print(f"Model input shape: {tuple(tensor1.shape)}")

    flow = run_inference(model, tensor1, tensor2)

    print(f"Model output flow shape: {flow.shape}")

    flow_original = resize_flow_to_original(
        flow,
        original_size,
    )

    output_directory = OUTPUT_ROOT / dataset_name

    save_results(
        dataset_name,
        flow_original,
        original_image1,
        original_image2,
        output_directory,
    )


def main():
    print("=" * 60)
    print("STAGE 5 - TASK 5: RAFT INFERENCE AND VISUALIZATION")
    print("=" * 60)

    model = load_trained_model()

    # Sintel: alley_1, frames 0001 and 0002.
    sintel_image1 = SINTEL_IMAGE_ROOT / "alley_1" / "frame_0001.png"
    sintel_image2 = SINTEL_IMAGE_ROOT / "alley_1" / "frame_0002.png"

    # KITTI: sample pair 000000.
    kitti_image1 = KITTI_IMAGE_ROOT / "000000_10.png"
    kitti_image2 = KITTI_IMAGE_ROOT / "000000_11.png"

    process_pair(
        model,
        "Sintel",
        sintel_image1,
        sintel_image2,
    )

    process_pair(
        model,
        "KITTI",
        kitti_image1,
        kitti_image2,
    )

    print("\n" + "=" * 60)
    print("TASK 5 INFERENCE COMPLETED")
    print(f"Results saved to: {OUTPUT_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
