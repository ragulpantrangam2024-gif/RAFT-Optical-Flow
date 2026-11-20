
from pathlib import Path
import sys
import time
import random
import csv
import io
from contextlib import redirect_stdout

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    Subset,
    WeightedRandomSampler
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_5_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(STAGE_5_DIR))

from task_02_raft_forward_pass import RAFTForwardModel
from task_03_raft_training_pipeline import (
    SintelDataset,
    KITTIDataset,
    sequence_loss
)

RESULT_DIR = (
    PROJECT_ROOT / "Stage_5" / "results" / "Task 4"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Training configuration
EPOCHS = 3
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5

UPDATE_ITERATIONS = 5
GAMMA = 0.8
GRADIENT_CLIP = 1.0

# Controlled CPU experiment
TRAIN_SAMPLES_PER_EPOCH = 100
VALIDATION_SAMPLES_PER_DATASET = 20

VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

NUM_WORKERS = 0

# Learning-rate scheduler
LR_STEP_SIZE = 1
LR_GAMMA = 0.8


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 3. DATA AUGMENTATION
# ============================================================

class AugmentedDataset(Dataset):
    """
    Applies spatial and brightness augmentation.

    Horizontal flip:
        Reverses image width and changes horizontal flow sign.

    Vertical flip:
        Reverses image height and changes vertical flow sign.

    Brightness:
        Applies the same brightness factor to both images.
    """

    def __init__(self, dataset, enabled=True):
        self.dataset = dataset
        self.enabled = enabled

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image1, image2, flow, valid, source = (
            self.dataset[index]
        )

        if not self.enabled:
            return image1, image2, flow, valid, source

        image1 = image1.clone()
        image2 = image2.clone()
        flow = flow.clone()
        valid = valid.clone()

        # Horizontal flip
        if random.random() < 0.5:

            image1 = torch.flip(image1, dims=[2])
            image2 = torch.flip(image2, dims=[2])
            flow = torch.flip(flow, dims=[2])
            valid = torch.flip(valid, dims=[1])

            flow[0] = -flow[0]

        # Vertical flip
        if random.random() < 0.1:

            image1 = torch.flip(image1, dims=[1])
            image2 = torch.flip(image2, dims=[1])
            flow = torch.flip(flow, dims=[1])
            valid = torch.flip(valid, dims=[0])

            flow[1] = -flow[1]

        # Shared brightness adjustment
        if random.random() < 0.5:

            brightness = random.uniform(0.8, 1.2)

            image1 = torch.clamp(
                image1 * brightness, 0.0, 1.0
            )

            image2 = torch.clamp(
                image2 * brightness, 0.0, 1.0
            )

        return image1, image2, flow, valid, source


# ============================================================
# 4. TRAINING AND VALIDATION SPLITS
# ============================================================

def create_dataset_splits(dataset, validation_count, seed):
    """
    Creates deterministic training and validation subsets.
    """

    dataset_size = len(dataset)

    if dataset_size < 2:
        raise ValueError(
            "Dataset needs at least two samples for splitting."
        )

    generator = torch.Generator().manual_seed(seed)

    indices = torch.randperm(
        dataset_size,
        generator=generator
    ).tolist()

    validation_count = min(
        validation_count,
        max(1, int(dataset_size * VALIDATION_SPLIT))
    )

    validation_indices = indices[:validation_count]
    training_indices = indices[validation_count:]

    return (
        Subset(dataset, training_indices),
        Subset(dataset, validation_indices)
    )


def prepare_datasets():

    print("\nLoading datasets...")

    sintel_dataset = SintelDataset()
    kitti_dataset = KITTIDataset()

    print(f"MPI-Sintel total samples: {len(sintel_dataset)}")
    print(f"KITTI total samples:      {len(kitti_dataset)}")

    if len(sintel_dataset) == 0:
        raise RuntimeError(
            "MPI-Sintel dataset is empty."
        )

    if len(kitti_dataset) == 0:
        raise RuntimeError(
            "KITTI dataset is empty."
        )

    sintel_train, sintel_val = create_dataset_splits(
        sintel_dataset,
        VALIDATION_SAMPLES_PER_DATASET,
        RANDOM_SEED
    )

    kitti_train, kitti_val = create_dataset_splits(
        kitti_dataset,
        VALIDATION_SAMPLES_PER_DATASET,
        RANDOM_SEED + 1
    )

    # Augmentation is applied only to training samples.
    sintel_train = AugmentedDataset(
        sintel_train,
        enabled=True
    )

    kitti_train = AugmentedDataset(
        kitti_train,
        enabled=True
    )

    sintel_val = AugmentedDataset(
        sintel_val,
        enabled=False
    )

    kitti_val = AugmentedDataset(
        kitti_val,
        enabled=False
    )

    train_dataset = ConcatDataset(
        [sintel_train, kitti_train]
    )

    validation_dataset = ConcatDataset(
        [sintel_val, kitti_val]
    )

    # Give each dataset equal total sampling weight.
    sintel_weight = 1.0 / len(sintel_train)
    kitti_weight = 1.0 / len(kitti_train)

    sample_weights = (
        [sintel_weight] * len(sintel_train)
        + [kitti_weight] * len(kitti_train)
    )

    sampler_generator = torch.Generator().manual_seed(
        RANDOM_SEED
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=TRAIN_SAMPLES_PER_EPOCH,
        replacement=True,
        generator=sampler_generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    print("\nDataset split information")
    print("-" * 45)

    print(f"Sintel training: {len(sintel_train)}")
    print(f"Sintel validation: {len(sintel_val)}")
    print(f"KITTI training: {len(kitti_train)}")
    print(f"KITTI validation: {len(kitti_val)}")

    print(
        f"Training batches per epoch: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(validation_loader)}"
    )

    return train_loader, validation_loader


# ============================================================
# 5. MODEL AND OPTIMIZER
# ============================================================

def create_model():

    model = RAFTForwardModel().to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = StepLR(
        optimizer,
        step_size=LR_STEP_SIZE,
        gamma=LR_GAMMA
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"\nModel parameters: {parameter_count:,}")

    return model, optimizer, scheduler


# ============================================================
# 6. FORWARD PASS WITHOUT ITERATION PRINTS
# ============================================================

def run_model(model, image1, image2):

    # The Task 2 model prints each iteration.
    # Suppress those messages during repeated training.
    with redirect_stdout(io.StringIO()):

        predictions = model(
            image1,
            image2,
            iterations=UPDATE_ITERATIONS
        )

    return predictions


# ============================================================
# 7. TRAINING FOR ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch
):

    model.train()

    running_loss = 0.0
    successful_batches = 0

    start_time = time.time()

    for batch_index, batch in enumerate(loader):

        image1, image2, ground_truth, valid_mask, source = batch

        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)
        valid_mask = valid_mask.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        predictions = run_model(
            model,
            image1,
            image2
        )

        loss = sequence_loss(
            predictions,
            ground_truth,
            valid_mask
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss in epoch {epoch}, "
                f"batch {batch_index + 1}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP
        )

        optimizer.step()

        running_loss += loss.item()
        successful_batches += 1

        if (
            (batch_index + 1) % 10 == 0
            or (batch_index + 1) == len(loader)
        ):

            average_loss = (
                running_loss / successful_batches
            )

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(loader)} | "
                f"Loss: {loss.item():.6f} | "
                f"Average: {average_loss:.6f} | "
                f"Source: {source[0]}"
            )

    elapsed = time.time() - start_time

    epoch_loss = (
        running_loss / max(successful_batches, 1)
    )

    return epoch_loss, elapsed


# ============================================================
# 8. VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader):

    model.eval()

    running_loss = 0.0
    successful_batches = 0

    for batch in loader:

        image1, image2, ground_truth, valid_mask, source = batch

        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)
        ground_truth = ground_truth.to(DEVICE)
        valid_mask = valid_mask.to(DEVICE)

        predictions = run_model(
            model,
            image1,
            image2
        )

        loss = sequence_loss(
            predictions,
            ground_truth,
            valid_mask
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite validation loss."
            )

        running_loss += loss.item()
        successful_batches += 1

    return running_loss / max(successful_batches, 1)


# ============================================================
# 9. SAVE CHECKPOINTS
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    validation_loss,
    best
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "input_size": (400, 120),
        "update_iterations": UPDATE_ITERATIONS,
    }

    latest_path = RESULT_DIR / "raft_latest_checkpoint.pth"

    torch.save(
        checkpoint,
        latest_path
    )

    if best:

        best_path = RESULT_DIR / "raft_best_model.pth"

        torch.save(
            checkpoint,
            best_path
        )

        print(f"Best model saved: {best_path}")


# ============================================================
# 10. SAVE TRAINING METRICS
# ============================================================

def save_metrics(history):

    csv_path = RESULT_DIR / "training_metrics.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "epoch",
            "training_loss",
            "validation_loss",
            "learning_rate",
            "epoch_time_seconds"
        ])

        writer.writerows(history)

    epochs = [row[0] for row in history]
    train_losses = [row[1] for row in history]
    validation_losses = [row[2] for row in history]

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training loss"
    )

    plt.plot(
        epochs,
        validation_losses,
        marker="s",
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Sequence loss")
    plt.title("RAFT Training and Validation Loss")

    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = RESULT_DIR / "training_loss_curve.png"

    plt.savefig(
        plot_path,
        dpi=200
    )

    plt.close()

    print(f"Metrics saved: {csv_path}")
    print(f"Loss curve saved: {plot_path}")


# ============================================================
# 11. MAIN TRAINING LOOP
# ============================================================

def main():

    print("=" * 65)
    print("STAGE 5 - TASK 4: RAFT MODEL TRAINING")
    print("=" * 65)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Training samples per epoch: {TRAIN_SAMPLES_PER_EPOCH}")

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Prepare datasets
    train_loader, validation_loader = prepare_datasets()

    # Create model
    model, optimizer, scheduler = create_model()

    history = []

    best_validation_loss = float("inf")

    total_start_time = time.time()

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print("\n" + "=" * 65)
        print(f"STARTING EPOCH {epoch}/{EPOCHS}")
        print("=" * 65)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Learning rate: {current_lr:.8f}")

        train_loss, epoch_time = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch
        )

        validation_loss = validate(
            model,
            validation_loader
        )

        scheduler.step()

        print("\nEpoch summary")
        print("-" * 45)

        print(f"Training loss: {train_loss:.6f}")
        print(f"Validation loss: {validation_loss:.6f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")

        history.append([
            epoch,
            train_loss,
            validation_loss,
            current_lr,
            epoch_time
        ])

        is_best = (
            validation_loss < best_validation_loss
        )

        if is_best:
            best_validation_loss = validation_loss

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            validation_loss,
            is_best
        )

        save_metrics(history)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total_time = time.time() - total_start_time

    summary_path = (
        RESULT_DIR / "training_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "STAGE 5 - TASK 4: RAFT MODEL TRAINING\n"
        )

        file.write("=" * 55 + "\n")

        file.write(f"Device: {DEVICE}\n")
        file.write(f"Epochs: {EPOCHS}\n")
        file.write(f"Batch size: {BATCH_SIZE}\n")
        file.write(f"Learning rate: {LEARNING_RATE}\n")

        file.write(
            f"Training samples per epoch: "
            f"{TRAIN_SAMPLES_PER_EPOCH}\n"
        )

        file.write(
            f"Validation batches: {len(validation_loader)}\n"
        )

        file.write(
            f"Best validation loss: "
            f"{best_validation_loss:.6f}\n"
        )

        file.write(
            f"Total training time: {total_time:.2f} seconds\n"
        )

        file.write("\nEpoch results\n")

        for row in history:

            file.write(
                f"Epoch {row[0]}: "
                f"train={row[1]:.6f}, "
                f"validation={row[2]:.6f}, "
                f"learning_rate={row[3]:.8f}, "
                f"time={row[4]:.2f}s\n"
            )

        file.write(
            "\nNote: This is an educational RAFT-style model "
            "trained from random initialization. "
            "These results are not directly comparable to "
            "the original RAFT benchmark.\n"
        )

    print("\n" + "=" * 65)
    print("RAFT TRAINING COMPLETED")
    print("=" * 65)

    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
