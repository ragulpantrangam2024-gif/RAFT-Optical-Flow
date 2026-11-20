
from pathlib import Path
import struct
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MPI-Sintel dataset
SINTEL_IMAGE_ROOT = (
    PROJECT_ROOT
    / "MPI-Sintel-training_images"
    / "training"
    / "final"
)

SINTEL_FLOW_ROOT = (
    PROJECT_ROOT
    / "training"
    / "flow"
)

SINTEL_INVALID_ROOT = (
    PROJECT_ROOT
    / "training"
    / "invalid"
)

# KITTI dataset
KITTI_ROOT = (
    PROJECT_ROOT
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
)

KITTI_IMAGE_ROOT = KITTI_ROOT / "image_2"
KITTI_FLOW_ROOT = KITTI_ROOT / "flow_occ"

# Model configuration
INPUT_WIDTH = 400
INPUT_HEIGHT = 120

BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

GAMMA = 0.8
UPDATE_ITERATIONS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "Stage_5"
    / "results"
    / "Task 3"
)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 2. IMPORT TASK 2 RAFT MODEL
# ============================================================

STAGE_5_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(STAGE_5_DIR))

from task_02_raft_forward_pass import RAFTForwardModel


# ============================================================
# 3. MPI-SINTEL .FLO READER
# ============================================================

def read_flo_file(path):
    """
    Read optical flow from MPI-Sintel .flo format.
    Returns an array with shape [H, W, 2].
    """

    with open(path, "rb") as file:

        magic = struct.unpack("<f", file.read(4))[0]

        if magic != 202021.25:
            raise ValueError(
                f"Invalid .flo file: {path}"
            )

        width = struct.unpack("<i", file.read(4))[0]
        height = struct.unpack("<i", file.read(4))[0]

        data = np.fromfile(file, dtype="<f4")

    if data.size != height * width * 2:
        raise ValueError(
            f"Unexpected flow data dimensions: {path}"
        )

    return data.reshape(height, width, 2)


# ============================================================
# 4. COMMON IMAGE AND FLOW RESIZING
# ============================================================

def resize_sample(image1, image2, flow, valid):
    """
    Resize image pair, ground-truth flow and validity mask.

    Flow vectors are scaled according to the image resizing.
    """

    original_height, original_width = flow.shape[:2]

    image1 = cv2.resize(
        image1,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    image2 = cv2.resize(
        image2,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    flow = cv2.resize(
        flow,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    valid = cv2.resize(
        valid.astype(np.uint8),
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    flow[..., 0] *= INPUT_WIDTH / original_width
    flow[..., 1] *= INPUT_HEIGHT / original_height

    return image1, image2, flow, valid


# ============================================================
# 5. CONVERT NUMPY DATA TO PYTORCH TENSORS
# ============================================================

def to_tensors(image1, image2, flow, valid):

    image1 = cv2.cvtColor(
        image1,
        cv2.COLOR_BGR2RGB
    )

    image2 = cv2.cvtColor(
        image2,
        cv2.COLOR_BGR2RGB
    )

    image1 = torch.from_numpy(
        image1.astype(np.float32) / 255.0
    ).permute(2, 0, 1)

    image2 = torch.from_numpy(
        image2.astype(np.float32) / 255.0
    ).permute(2, 0, 1)

    flow = torch.from_numpy(
        flow.astype(np.float32)
    ).permute(2, 0, 1)

    valid = torch.from_numpy(
        valid.astype(np.bool_)
    )

    return image1, image2, flow, valid


# ============================================================
# 6. MPI-SINTEL DATASET
# ============================================================

class SintelDataset(Dataset):

    def __init__(self):

        self.samples = []

        if not SINTEL_IMAGE_ROOT.exists():
            print(
                f"WARNING: Sintel image folder not found: "
                f"{SINTEL_IMAGE_ROOT}"
            )
            return

        for scene_dir in sorted(
            SINTEL_IMAGE_ROOT.iterdir()
        ):

            if not scene_dir.is_dir():
                continue

            image_paths = sorted(
                scene_dir.glob("*.png")
            )

            for index in range(len(image_paths) - 1):

                image1_path = image_paths[index]
                image2_path = image_paths[index + 1]

                flow_path = (
                    SINTEL_FLOW_ROOT
                    / scene_dir.name
                    / f"{image1_path.stem}.flo"
                )

                if flow_path.exists():

                    self.samples.append(
                        (
                            image1_path,
                            image2_path,
                            flow_path,
                            scene_dir.name
                        )
                    )

        print(
            f"Sintel Final: found {len(self.samples)} "
            f"valid image pairs."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image1_path, image2_path, flow_path, scene = (
            self.samples[index]
        )

        image1 = cv2.imread(
            str(image1_path)
        )

        image2 = cv2.imread(
            str(image2_path)
        )

        if image1 is None or image2 is None:
            raise RuntimeError(
                "Could not read Sintel image pair."
            )

        flow = read_flo_file(flow_path)

        valid = (
            np.isfinite(flow).all(axis=2)
        )

        invalid_path = (
            SINTEL_INVALID_ROOT
            / scene
            / f"{image1_path.stem}.png"
        )

        if invalid_path.exists():

            invalid_mask = cv2.imread(
                str(invalid_path),
                cv2.IMREAD_GRAYSCALE
            )

            if invalid_mask is not None:

                valid &= invalid_mask == 0

        valid &= (
            np.linalg.norm(flow, axis=2) < 1000
        )

        image1, image2, flow, valid = resize_sample(
            image1,
            image2,
            flow,
            valid
        )

        return to_tensors(
            image1,
            image2,
            flow,
            valid
        ) + ("MPI-Sintel",)


# ============================================================
# 7. KITTI DATASET
# ============================================================

class KITTIDataset(Dataset):

    def __init__(self):

        self.samples = []

        if not KITTI_IMAGE_ROOT.exists():
            print(
                f"WARNING: KITTI image folder not found: "
                f"{KITTI_IMAGE_ROOT}"
            )
            return

        for image1_path in sorted(
            KITTI_IMAGE_ROOT.glob("*_10.png")
        ):

            image2_path = image1_path.with_name(
                image1_path.name.replace(
                    "_10.png",
                    "_11.png"
                )
            )

            flow_path = (
                KITTI_FLOW_ROOT / image1_path.name
            )

            if (
                image2_path.exists()
                and flow_path.exists()
            ):

                self.samples.append(
                    (
                        image1_path,
                        image2_path,
                        flow_path
                    )
                )

        print(
            f"KITTI: found {len(self.samples)} "
            f"valid image pairs."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image1_path, image2_path, flow_path = (
            self.samples[index]
        )

        image1 = cv2.imread(
            str(image1_path)
        )

        image2 = cv2.imread(
            str(image2_path)
        )

        flow_png = cv2.imread(
            str(flow_path),
            cv2.IMREAD_UNCHANGED
        )

        if (
            image1 is None
            or image2 is None
            or flow_png is None
        ):

            raise RuntimeError(
                "Could not read KITTI sample."
            )

        # Decode KITTI flow PNG.
        # OpenCV loads the PNG in BGR channel order.

        flow = np.zeros(
            (
                flow_png.shape[0],
                flow_png.shape[1],
                2
            ),
            dtype=np.float32
        )

        flow[..., 0] = (
            flow_png[..., 2].astype(np.float32)
            - 32768.0
        ) / 64.0

        flow[..., 1] = (
            flow_png[..., 1].astype(np.float32)
            - 32768.0
        ) / 64.0

        valid = flow_png[..., 0] > 0

        image1, image2, flow, valid = resize_sample(
            image1,
            image2,
            flow,
            valid
        )

        return to_tensors(
            image1,
            image2,
            flow,
            valid
        ) + ("KITTI",)


# ============================================================
# 8. RAFT SEQUENCE LOSS
# ============================================================

def sequence_loss(
    predictions,
    ground_truth,
    valid_mask
):
    """
    Weighted loss across all recurrent flow predictions.
    """

    total_loss = 0.0

    prediction_count = len(predictions)

    valid = valid_mask.bool()

    valid_count = valid.sum()

    if valid_count.item() == 0:
        raise ValueError(
            "Batch contains no valid ground-truth pixels."
        )

    for index, prediction in enumerate(predictions):

        weight = GAMMA ** (
            prediction_count - index - 1
        )

        error = torch.abs(
            prediction - ground_truth
        )

        masked_error = (
            error * valid.unsqueeze(1)
        )

        step_loss = masked_error.sum() / (
            valid_count * 2
        )

        total_loss += weight * step_loss

    return total_loss


# ============================================================
# 9. MAIN TRAINING PIPELINE TEST
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 5 - TASK 3: RAFT TRAINING PIPELINE")
    print("=" * 65)

    print(f"Device: {DEVICE}")

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATASETS
    # --------------------------------------------------------

    sintel_dataset = SintelDataset()
    kitti_dataset = KITTIDataset()

    print("\nDataset information")
    print("-" * 40)

    print(
        f"MPI-Sintel samples: {len(sintel_dataset)}"
    )

    print(
        f"KITTI samples:      {len(kitti_dataset)}"
    )

    datasets = []

    if len(sintel_dataset) > 0:
        datasets.append(sintel_dataset)

    if len(kitti_dataset) > 0:
        datasets.append(kitti_dataset)

    if not datasets:
        raise RuntimeError(
            "No training samples found. "
            "Check the dataset paths."
        )

    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)

    print(
        f"Combined samples:   {len(combined_dataset)}"
    )

    train_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    # --------------------------------------------------------
    # INITIALIZE MODEL
    # --------------------------------------------------------

    model = RAFTForwardModel().to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nModel parameters: {parameter_count:,}"
    )

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Update iterations: {UPDATE_ITERATIONS}")

    # --------------------------------------------------------
    # ONE-BATCH TRAINING TEST
    # --------------------------------------------------------

    print("\nRunning one-batch training test...")

    model.train()

    start_time = time.time()

    image1, image2, ground_truth, valid_mask, source = (
        next(iter(train_loader))
    )

    image1 = image1.to(DEVICE)
    image2 = image2.to(DEVICE)
    ground_truth = ground_truth.to(DEVICE)
    valid_mask = valid_mask.to(DEVICE)

    print(f"Batch source: {source}")
    print(f"Image shape: {tuple(image1.shape)}")
    print(
        f"Ground-truth flow shape: "
        f"{tuple(ground_truth.shape)}"
    )

    print(
        f"Valid pixels: {int(valid_mask.sum())}"
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    predictions = model(
        image1,
        image2,
        iterations=UPDATE_ITERATIONS
    )

    loss = sequence_loss(
        predictions,
        ground_truth,
        valid_mask
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite training loss: {loss.item()}"
        )

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if not gradients_exist:
        raise RuntimeError(
            "No gradients were generated."
        )

    optimizer.step()
    scheduler.step()

    elapsed_time = time.time() - start_time

    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = (
        RESULT_DIR
        / "task3_pipeline_checkpoint.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": float(loss.item()),
            "model_parameters": parameter_count,
            "input_size": (
                INPUT_WIDTH,
                INPUT_HEIGHT
            ),
            "iterations": UPDATE_ITERATIONS,
            "batch_source": source[0],
        },
        checkpoint_path
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    summary_path = (
        RESULT_DIR
        / "training_pipeline_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STAGE 5 - TASK 3: RAFT TRAINING PIPELINE\n"
        )

        file.write("=" * 55 + "\n")

        file.write(f"Device: {DEVICE}\n")

        file.write(
            f"MPI-Sintel samples: {len(sintel_dataset)}\n"
        )

        file.write(
            f"KITTI samples: {len(kitti_dataset)}\n"
        )

        file.write(
            f"Combined samples: {len(combined_dataset)}\n"
        )

        file.write(
            f"Model parameters: {parameter_count:,}\n"
        )

        file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        file.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )

        file.write(
            f"Update iterations: {UPDATE_ITERATIONS}\n"
        )

        file.write(
            f"Batch source: {source[0]}\n"
        )

        file.write(
            f"Valid pixels: {int(valid_mask.sum())}\n"
        )

        file.write(
            f"Sequence loss: {loss.item():.6f}\n"
        )

        file.write(
            f"Gradient norm: {float(gradient_norm):.6f}\n"
        )

        file.write(
            f"Gradients generated: {gradients_exist}\n"
        )

        file.write(
            f"Training step time: {elapsed_time:.4f} seconds\n"
        )

        file.write(
            f"Checkpoint: {checkpoint_path.name}\n"
        )

        file.write(
            "\nThis task validates one training step only. "
            "Full model training is reserved for Stage 5 Task 4.\n"
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\nTraining pipeline test completed.")

    print(
        f"Sequence loss: {loss.item():.6f}"
    )

    print(
        f"Gradient norm: {float(gradient_norm):.6f}"
    )

    print(
        f"Gradients generated: {gradients_exist}"
    )

    print(
        f"Training step time: {elapsed_time:.4f} seconds"
    )

    print(
        f"\nCheckpoint saved: {checkpoint_path}"
    )

    print(
        f"Summary saved: {summary_path}"
    )


if __name__ == "__main__":
    main()
