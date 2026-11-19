
"""
============================================================================
STAGE 4 - TASK 1: CNN FUNDAMENTALS USING PYTORCH
============================================================================

Objective:
Understand how a Convolutional Neural Network processes an image.

Concepts:
1. Image to PyTorch Tensor
2. Convolution
3. ReLU Activation
4. Max Pooling
5. Feature Map Extraction
6. CNN Architecture

Dataset:
KITTI 2015 Optical Flow Training Images

Note:
The CNN is randomly initialized. It demonstrates CNN operations and
feature-map generation, not learned optical-flow features.
============================================================================
"""

from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Stage_4"
    / "results"
    / "Task 1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = DATASET_ROOT / "image_2" / "000000_10.png"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("=" * 75)
print("STAGE 4 - TASK 1: CNN FUNDAMENTALS")
print("=" * 75)

print(f"\nProject directory: {PROJECT_ROOT}")
print(f"Dataset directory: {DATASET_ROOT}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Processing device: {DEVICE}")


# ============================================================================
# 2. LOAD KITTI IMAGE
# ============================================================================

def load_kitti_image(image_path):
    """
    Load a KITTI image and convert it from BGR to RGB.
    """

    if not image_path.exists():

        raise FileNotFoundError(
            f"KITTI image not found:\n{image_path}\n"
            "Check the dataset location and image filename."
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError("Unable to load the input image.")

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


print("\nLoading KITTI image...")

image = load_kitti_image(IMAGE_PATH)

height, width, channels = image.shape

print(f"Image shape: {image.shape}")
print(f"Image height: {height}")
print(f"Image width: {width}")
print(f"Image channels: {channels}")


# ============================================================================
# 3. IMAGE TO PYTORCH TENSOR
# ============================================================================

print("\nConverting image into PyTorch tensor...")

# Convert uint8 image to float32 and normalize to [0, 1].
image_float = image.astype(np.float32) / 255.0

# HWC -> CHW
image_tensor = torch.from_numpy(
    image_float
).permute(2, 0, 1)

# Add batch dimension: CHW -> NCHW
image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)

print(f"Tensor shape: {tuple(image_tensor.shape)}")
print(f"Tensor data type: {image_tensor.dtype}")
print(f"Tensor minimum: {image_tensor.min().item():.4f}")
print(f"Tensor maximum: {image_tensor.max().item():.4f}")


# ============================================================================
# 4. DEFINE CNN ARCHITECTURE
# ============================================================================

class SimpleCNN(nn.Module):
    """
    A small CNN designed to demonstrate convolution,
    ReLU activation, pooling and feature extraction.
    """

    def __init__(self):

        super(SimpleCNN, self).__init__()

        # ------------------------------------------------------------
        # CONVOLUTION BLOCK 1
        # Input channels: 3
        # Output channels: 16
        # ------------------------------------------------------------

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu1 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ------------------------------------------------------------
        # CONVOLUTION BLOCK 2
        # Input channels: 16
        # Output channels: 32
        # ------------------------------------------------------------

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ------------------------------------------------------------
        # CONVOLUTION BLOCK 3
        # Input channels: 32
        # Output channels: 64
        # ------------------------------------------------------------

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.relu3 = nn.ReLU()

    def forward(self, x):

        feature_maps = {}

        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)

        feature_maps["conv1"] = x

        x = self.pool1(x)

        feature_maps["pool1"] = x

        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)

        feature_maps["conv2"] = x

        x = self.pool2(x)

        feature_maps["pool2"] = x

        # Block 3
        x = self.conv3(x)
        x = self.relu3(x)

        feature_maps["conv3"] = x

        return feature_maps


# ============================================================================
# 5. INITIALIZE CNN
# ============================================================================

print("\nInitializing CNN...")

model = SimpleCNN().to(DEVICE)

model.eval()

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")


# ============================================================================
# 6. FORWARD PASS
# ============================================================================

print("\nPerforming CNN forward pass...")

start_time = time.perf_counter()

with torch.inference_mode():

    feature_maps = model(image_tensor)

inference_time = time.perf_counter() - start_time

print(f"Forward pass completed in {inference_time:.4f} seconds")


# ============================================================================
# 7. DISPLAY FEATURE MAP DIMENSIONS
# ============================================================================

print("\nFEATURE MAP DIMENSIONS")
print("-" * 75)

for layer_name, feature in feature_maps.items():

    print(
        f"{layer_name:10s} "
        f"Shape: {tuple(feature.shape)}"
    )


# ============================================================================
# 8. SAVE INPUT IMAGE
# ============================================================================

plt.figure(figsize=(12, 5))

plt.imshow(image)

plt.title("Original KITTI Input Image")

plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "original_kitti_image.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================================
# 9. VISUALIZE FEATURE MAPS
# ============================================================================

def visualize_feature_maps(
    feature_tensor,
    layer_name,
    output_path,
    number_of_maps=8
):
    """
    Visualize the first eight channels of a CNN feature tensor.
    """

    feature_array = (
        feature_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    number_of_channels = feature_array.shape[0]

    number_of_maps = min(
        number_of_maps,
        number_of_channels
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 7)
    )

    axes = axes.flatten()

    for index in range(number_of_maps):

        axes[index].imshow(
            feature_array[index],
            cmap="viridis"
        )

        axes[index].set_title(
            f"Feature Map {index + 1}"
        )

        axes[index].axis("off")

    for index in range(number_of_maps, len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        f"{layer_name} - CNN Feature Maps",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path.name}")


print("\nGenerating feature-map visualizations...")

visualize_feature_maps(
    feature_maps["conv1"],
    "Convolution Layer 1",
    OUTPUT_DIR / "conv1_feature_maps.png"
)

visualize_feature_maps(
    feature_maps["conv2"],
    "Convolution Layer 2",
    OUTPUT_DIR / "conv2_feature_maps.png"
)

visualize_feature_maps(
    feature_maps["conv3"],
    "Convolution Layer 3",
    OUTPUT_DIR / "conv3_feature_maps.png"
)


# ============================================================================
# 10. VISUALIZE CNN ARCHITECTURE OUTPUTS
# ============================================================================

layer_names = list(feature_maps.keys())

fig, axes = plt.subplots(
    1,
    len(layer_names),
    figsize=(18, 5)
)

for axis, layer_name in zip(axes, layer_names):

    feature = feature_maps[layer_name][0, 0]

    axis.imshow(
        feature.detach().cpu().numpy(),
        cmap="viridis"
    )

    axis.set_title(layer_name)

    axis.axis("off")

fig.suptitle(
    "CNN Feature Extraction Across Layers",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cnn_layer_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("Saved: cnn_layer_comparison.png")


# ============================================================================
# 11. CALCULATE FEATURE MAP STATISTICS
# ============================================================================

print("\nFEATURE MAP STATISTICS")
print("-" * 75)

statistics = {}

for layer_name, feature in feature_maps.items():

    statistics[layer_name] = {

        "shape": tuple(feature.shape),

        "mean": feature.mean().item(),

        "std": feature.std().item(),

        "minimum": feature.min().item(),

        "maximum": feature.max().item()
    }

    values = statistics[layer_name]

    print(f"\n{layer_name}")

    print(f"Mean: {values['mean']:.6f}")

    print(f"Standard deviation: {values['std']:.6f}")

    print(f"Minimum: {values['minimum']:.6f}")

    print(f"Maximum: {values['maximum']:.6f}")


# ============================================================================
# 12. SAVE SUMMARY
# ============================================================================

summary_path = OUTPUT_DIR / "cnn_fundamentals_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 4 - TASK 1: CNN FUNDAMENTALS\n")

    file.write("=" * 65 + "\n\n")

    file.write("INPUT IMAGE\n")

    file.write(f"Image path: {IMAGE_PATH}\n")

    file.write(f"Image dimensions: {image.shape}\n")

    file.write(f"Tensor shape: {tuple(image_tensor.shape)}\n")

    file.write(f"Device: {DEVICE}\n\n")

    file.write("CNN ARCHITECTURE\n")

    file.write("Conv1: 3 -> 16 channels\n")

    file.write("ReLU1\n")

    file.write("MaxPool1: 2x2\n")

    file.write("Conv2: 16 -> 32 channels\n")

    file.write("ReLU2\n")

    file.write("MaxPool2: 2x2\n")

    file.write("Conv3: 32 -> 64 channels\n")

    file.write("ReLU3\n\n")

    file.write("MODEL PARAMETERS\n")

    file.write(f"Total parameters: {total_parameters}\n")

    file.write(
        f"Trainable parameters: {trainable_parameters}\n"
    )

    file.write(
        f"Forward pass runtime: {inference_time:.6f} seconds\n\n"
    )

    file.write("FEATURE MAP DETAILS\n")

    for layer_name, values in statistics.items():

        file.write(f"\n{layer_name}\n")

        file.write(f"Shape: {values['shape']}\n")

        file.write(f"Mean: {values['mean']:.6f}\n")

        file.write(f"Standard deviation: {values['std']:.6f}\n")

        file.write(f"Minimum: {values['minimum']:.6f}\n")

        file.write(f"Maximum: {values['maximum']:.6f}\n")

    file.write("\nIMPORTANT NOTE\n")

    file.write(
        "The CNN uses randomly initialized weights. "
        "The feature maps demonstrate convolution operations "
        "and are not learned optical-flow features.\n"
    )


# ============================================================================
# 13. COMPLETION
# ============================================================================

print("\n" + "=" * 75)

print("STAGE 4 - TASK 1 COMPLETED SUCCESSFULLY")

print("=" * 75)

print(f"\nResults saved to:\n{OUTPUT_DIR}")

print("\nGenerated files:")

for file_path in sorted(OUTPUT_DIR.iterdir()):

    if file_path.is_file():

        print(f" - {file_path.name}")

print("\nNext: Inspect the feature maps and understand each CNN layer.")
