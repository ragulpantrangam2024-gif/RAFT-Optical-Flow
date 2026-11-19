"""
STAGE 4 - TASK 2: CNN FEATURE EXTRACTION

Purpose:
1. Load two consecutive KITTI images.
2. Process both images through the same CNN feature encoder.
3. Extract multi-level feature maps.
4. Compare corresponding feature representations.
5. Visualize feature maps, cosine similarity and feature differences.
6. Save numerical results and visualizations.

Note:
The CNN is randomly initialized. This task demonstrates the feature
extraction pipeline; it does not train an optical-flow network.
"""

from pathlib import Path
import time
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"D:\rwu\rwu jobs\Computer_Vision_Projects\RAFT_Optical_Flow"
)

DATASET_DIR = (
    PROJECT_DIR
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
)

IMAGE1_PATH = DATASET_DIR / "image_2" / "000000_10.png"
IMAGE2_PATH = DATASET_DIR / "image_2" / "000000_11.png"

OUTPUT_DIR = PROJECT_DIR / "Stage_4" / "results" / "Task 2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of feature maps to display from each layer.
FEATURE_MAPS_TO_DISPLAY = 8

# Reproducibility for the randomly initialized CNN.
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# 2. CNN FEATURE ENCODER
# ============================================================

class CNNFeatureEncoder(nn.Module):
    """
    A small CNN encoder for extracting image features.

    Both KITTI images will use the SAME encoder instance.
    """

    def __init__(self):
        super().__init__()

        # First feature extraction block.
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # Downsampling by a factor of 2.
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Second feature extraction block.
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

        # Second downsampling.
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Third feature extraction block.
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        features = {}

        features["conv1"] = self.block1(x)

        features["pool1"] = self.pool1(
            features["conv1"]
        )

        features["conv2"] = self.block2(
            features["pool1"]
        )

        features["pool2"] = self.pool2(
            features["conv2"]
        )

        features["conv3"] = self.block3(
            features["pool2"]
        )

        return features


# ============================================================
# 3. IMAGE LOADING AND PREPROCESSING
# ============================================================

def load_image(image_path):

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # OpenCV loads BGR. Convert to RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


def image_to_tensor(image):

    # Convert uint8 image into float32 [0, 1].
    image_float = image.astype(np.float32) / 255.0

    # HWC -> CHW.
    tensor = torch.from_numpy(
        image_float.transpose(2, 0, 1)
    )

    # Add batch dimension: CHW -> NCHW.
    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# ============================================================
# 4. FEATURE MAP VISUALIZATION
# ============================================================

def visualize_feature_maps(
    feature_tensor,
    title,
    output_path,
    number_of_maps=8
):

    feature_array = (
        feature_tensor[0]
        .detach()
        .cpu()
        .numpy()
    )

    total_channels = feature_array.shape[0]

    number_of_maps = min(
        number_of_maps,
        total_channels
    )

    columns = 4
    rows = int(np.ceil(number_of_maps / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 4 * rows)
    )

    axes = np.array(axes).reshape(-1)

    for index in range(number_of_maps):

        feature_map = feature_array[index]

        axes[index].imshow(
            feature_map,
            cmap="viridis"
        )

        axes[index].set_title(
            f"Feature Map {index + 1}"
        )

        axes[index].axis("off")

    for index in range(number_of_maps, len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path.name}")


# ============================================================
# 5. FEATURE COMPARISON
# ============================================================

def calculate_feature_comparison(
    features1,
    features2
):

    comparison_results = {}

    for layer_name in features1:

        feature1 = features1[layer_name]
        feature2 = features2[layer_name]

        # Absolute difference between corresponding features.
        difference = torch.abs(
            feature1 - feature2
        )

        # Cosine similarity across channels at each spatial position.
        similarity = F.cosine_similarity(
            feature1,
            feature2,
            dim=1,
            eps=1e-8
        )

        comparison_results[layer_name] = {
            "difference": difference,
            "similarity": similarity
        }

    return comparison_results


# ============================================================
# 6. SAVE FEATURE COMPARISON VISUALIZATION
# ============================================================

def save_comparison_visualization(
    comparison_results
):

    layer_names = list(comparison_results.keys())

    fig, axes = plt.subplots(
        len(layer_names),
        2,
        figsize=(14, 4 * len(layer_names))
    )

    for row, layer_name in enumerate(layer_names):

        difference = (
            comparison_results[layer_name]["difference"]
            .mean(dim=1)[0]
            .detach()
            .cpu()
            .numpy()
        )

        similarity = (
            comparison_results[layer_name]["similarity"][0]
            .detach()
            .cpu()
            .numpy()
        )

        axes[row, 0].imshow(
            difference,
            cmap="inferno"
        )

        axes[row, 0].set_title(
            f"{layer_name} - Mean Absolute Feature Difference"
        )

        axes[row, 1].imshow(
            similarity,
            cmap="viridis",
            vmin=-1,
            vmax=1
        )

        axes[row, 1].set_title(
            f"{layer_name} - Cosine Similarity"
        )

        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / "feature_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path.name}")


# ============================================================
# 7. SAVE FEATURE STATISTICS
# ============================================================

def save_feature_statistics(
    features1,
    features2,
    comparison_results
):

    csv_path = (
        OUTPUT_DIR / "feature_extraction_metrics.csv"
    )

    rows = []

    for layer_name in features1:

        feature1 = features1[layer_name].detach()
        feature2 = features2[layer_name].detach()

        difference = (
            comparison_results[layer_name]["difference"]
            .detach()
        )

        similarity = (
            comparison_results[layer_name]["similarity"]
            .detach()
        )

        rows.append({
            "layer": layer_name,
            "channels": feature1.shape[1],
            "height": feature1.shape[2],
            "width": feature1.shape[3],
            "image1_mean": feature1.mean().item(),
            "image1_std": feature1.std().item(),
            "image2_mean": feature2.mean().item(),
            "image2_std": feature2.std().item(),
            "mean_absolute_difference": difference.mean().item(),
            "mean_cosine_similarity": similarity.mean().item()
        })

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {csv_path.name}")

    return rows


# ============================================================
# 8. SAVE SUMMARY
# ============================================================

def save_summary(
    image1,
    image2,
    encoder,
    features1,
    features2,
    comparison_rows,
    runtime
):

    summary_path = (
        OUTPUT_DIR / "cnn_feature_extraction_summary.txt"
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )

    lines = []

    lines.append("=" * 70)
    lines.append("STAGE 4 - TASK 2: CNN FEATURE EXTRACTION")
    lines.append("=" * 70)

    lines.append(f"Processing device: {DEVICE}")
    lines.append(f"Image 1: {IMAGE1_PATH.name}")
    lines.append(f"Image 2: {IMAGE2_PATH.name}")

    lines.append(
        f"Image dimensions: {image1.shape[1]} x {image1.shape[0]}"
    )

    lines.append(
        f"Total CNN parameters: {total_parameters:,}"
    )

    lines.append(
        f"Feature extraction runtime: {runtime:.6f} seconds"
    )

    lines.append("")
    lines.append("FEATURE EXTRACTION DIMENSIONS")
    lines.append("-" * 70)

    for layer_name in features1:

        lines.append(
            f"{layer_name}: "
            f"{tuple(features1[layer_name].shape)}"
        )

    lines.append("")
    lines.append("FEATURE COMPARISON STATISTICS")
    lines.append("-" * 70)

    for row in comparison_rows:

        lines.append(f"\nLayer: {row['layer']}")

        lines.append(
            f"Channels: {row['channels']}"
        )

        lines.append(
            f"Spatial dimensions: "
            f"{row['height']} x {row['width']}"
        )

        lines.append(
            f"Mean absolute feature difference: "
            f"{row['mean_absolute_difference']:.6f}"
        )

        lines.append(
            f"Mean cosine similarity: "
            f"{row['mean_cosine_similarity']:.6f}"
        )

    lines.append("")
    lines.append("IMPORTANT NOTE")
    lines.append(
        "The CNN uses randomly initialized weights. "
        "The extracted representations are not trained optical-flow features."
    )

    lines.append(
        "Both images are processed using the same CNN encoder instance."
    )

    lines.append("=" * 70)
    lines.append("TASK 2 COMPLETED SUCCESSFULLY")
    lines.append("=" * 70)

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(lines))

    print(f"Saved: {summary_path.name}")


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 4 - TASK 2: CNN FEATURE EXTRACTION")
    print("=" * 70)

    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Processing device: {DEVICE}")

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print("\nLoading KITTI images...")

    image1 = load_image(IMAGE1_PATH)
    image2 = load_image(IMAGE2_PATH)

    if image1.shape != image2.shape:
        raise ValueError(
            "The two images must have matching dimensions. "
            f"Found {image1.shape} and {image2.shape}."
        )

    print(f"Image 1 shape: {image1.shape}")
    print(f"Image 2 shape: {image2.shape}")

    # --------------------------------------------------------
    # Convert images into tensors
    # --------------------------------------------------------

    print("\nConverting images into PyTorch tensors...")

    tensor1 = image_to_tensor(image1)
    tensor2 = image_to_tensor(image2)

    print(f"Image 1 tensor: {tuple(tensor1.shape)}")
    print(f"Image 2 tensor: {tuple(tensor2.shape)}")

    # --------------------------------------------------------
    # Save original images
    # --------------------------------------------------------

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("KITTI Frame 1")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title("KITTI Frame 2")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "input_image_pair.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Initialize CNN encoder
    # --------------------------------------------------------

    print("\nInitializing CNN feature encoder...")

    encoder = CNNFeatureEncoder().to(DEVICE)
    encoder.eval()

    total_parameters = sum(
        parameter.numel()
        for parameter in encoder.parameters()
    )

    print(
        f"Total parameters: {total_parameters:,}"
    )

    # --------------------------------------------------------
    # Extract features using shared encoder
    # --------------------------------------------------------

    print("\nExtracting features from both frames...")

    start_time = time.perf_counter()

    with torch.inference_mode():

        features1 = encoder(tensor1)
        features2 = encoder(tensor2)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    runtime = time.perf_counter() - start_time

    print(
        f"Feature extraction completed in {runtime:.4f} seconds"
    )

    # --------------------------------------------------------
    # Print feature dimensions
    # --------------------------------------------------------

    print("\nFEATURE MAP DIMENSIONS")
    print("-" * 70)

    for layer_name in features1:

        print(
            f"{layer_name:<10} "
            f"Frame 1: {tuple(features1[layer_name].shape)} "
            f"| Frame 2: {tuple(features2[layer_name].shape)}"
        )

    # --------------------------------------------------------
    # Visualize feature maps
    # --------------------------------------------------------

    print("\nGenerating feature-map visualizations...")

    for layer_name in features1:

        visualize_feature_maps(
            features1[layer_name],
            f"Frame 1 - {layer_name} Feature Maps",
            OUTPUT_DIR / f"frame1_{layer_name}_features.png",
            FEATURE_MAPS_TO_DISPLAY
        )

        visualize_feature_maps(
            features2[layer_name],
            f"Frame 2 - {layer_name} Feature Maps",
            OUTPUT_DIR / f"frame2_{layer_name}_features.png",
            FEATURE_MAPS_TO_DISPLAY
        )

    # --------------------------------------------------------
    # Compare extracted features
    # --------------------------------------------------------

    print("\nComparing feature representations...")

    comparison_results = calculate_feature_comparison(
        features1,
        features2
    )

    save_comparison_visualization(
        comparison_results
    )

    # --------------------------------------------------------
    # Save numerical metrics
    # --------------------------------------------------------

    print("\nSaving feature statistics...")

    comparison_rows = save_feature_statistics(
        features1,
        features2,
        comparison_results
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        image1,
        image2,
        encoder,
        features1,
        features2,
        comparison_rows,
        runtime
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STAGE 4 - TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nResults saved to:\n{OUTPUT_DIR}")

    print("\nGenerated files include:")
    print(" - input_image_pair.png")
    print(" - frame1_conv1_features.png")
    print(" - frame2_conv1_features.png")
    print(" - frame1_conv2_features.png")
    print(" - frame2_conv2_features.png")
    print(" - frame1_conv3_features.png")
    print(" - frame2_conv3_features.png")
    print(" - feature_comparison.png")
    print(" - feature_extraction_metrics.csv")
    print(" - cnn_feature_extraction_summary.txt")


if __name__ == "__main__":
    main()
