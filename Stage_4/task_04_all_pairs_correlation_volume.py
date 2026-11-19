
"""
STAGE 4 - TASK 4: ALL-PAIRS CORRELATION VOLUME

Builds a multi-level correlation pyramid from CNN feature maps
of two consecutive KITTI images.

Educational implementation inspired by the correlation operation
used in RAFT. The CNN is randomly initialized, not pretrained.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

OUTPUT_DIR = PROJECT_DIR / "Stage_4" / "results" / "Task 4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_HEIGHT = 24
GRID_WIDTH = 40
PYRAMID_LEVELS = 4

# Fixed seed makes this task reproducible.
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 2. CNN FEATURE ENCODER
# Same architecture as Task 3
# ============================================================

class CNNFeatureEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x


# ============================================================
# 3. IMAGE LOADING AND PREPROCESSING
# ============================================================

print("=" * 72)
print("STAGE 4 - TASK 4: ALL-PAIRS CORRELATION VOLUME")
print("=" * 72)

print("\nLoading KITTI image pair...")

image1 = cv2.imread(str(IMAGE1_PATH))
image2 = cv2.imread(str(IMAGE2_PATH))

if image1 is None or image2 is None:
    raise FileNotFoundError(
        "Could not load KITTI images. Check the dataset paths."
    )

if image1.shape != image2.shape:
    raise ValueError("Input images must have identical dimensions.")

height, width = image1.shape[:2]

print("Image dimensions:", width, "x", height)
print("Processing device:", DEVICE)


def preprocess_image(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb.copy())
    tensor = tensor.permute(2, 0, 1).float() / 255.0

    return tensor.unsqueeze(0)


tensor1 = preprocess_image(image1).to(DEVICE)
tensor2 = preprocess_image(image2).to(DEVICE)


# ============================================================
# 4. CNN FEATURE EXTRACTION
# ============================================================

print("\nInitializing CNN feature encoder...")

model = CNNFeatureEncoder().to(DEVICE)
model.eval()

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

print("CNN parameters:", parameter_count)

start_time = time.perf_counter()

with torch.no_grad():

    features1 = model(tensor1)
    features2 = model(tensor2)

    features1 = F.adaptive_avg_pool2d(
        features1, (GRID_HEIGHT, GRID_WIDTH)
    )

    features2 = F.adaptive_avg_pool2d(
        features2, (GRID_HEIGHT, GRID_WIDTH)
    )

feature_time = time.perf_counter() - start_time

print("Feature extraction time: {:.4f} seconds".format(feature_time))
print("Feature 1 shape:", tuple(features1.shape))
print("Feature 2 shape:", tuple(features2.shape))


# ============================================================
# 5. NORMALIZE FEATURE VECTORS
# ============================================================

print("\nNormalizing feature vectors...")

# [1, 64, 24, 40] -> [1, 960, 64]

feature1 = features1.flatten(2).transpose(1, 2)
feature2 = features2.flatten(2).transpose(1, 2)

feature1 = F.normalize(feature1, p=2, dim=-1)
feature2 = F.normalize(feature2, p=2, dim=-1)

number_of_locations = GRID_HEIGHT * GRID_WIDTH

print("Feature locations per image:", number_of_locations)
print("Feature channels:", feature1.shape[-1])


# ============================================================
# 6. COMPUTE BASE ALL-PAIRS CORRELATION
# ============================================================

print("\nComputing base all-pairs correlation...")

start_time = time.perf_counter()

with torch.no_grad():

    # [1, 960, 64] x [1, 64, 960]
    # Result: [1, 960, 960]

    correlation = torch.matmul(
        feature1,
        feature2.transpose(1, 2)
    )

    # Reshape into a 4D spatial correlation volume:
    # [B, H1, W1, H2, W2]

    correlation_volume = correlation.reshape(
        1,
        GRID_HEIGHT,
        GRID_WIDTH,
        GRID_HEIGHT,
        GRID_WIDTH
    )

correlation_time = time.perf_counter() - start_time

base_np = correlation_volume.squeeze(0).cpu().numpy()

print("Base correlation volume shape:", tuple(correlation_volume.shape))
print("Correlation computation time: {:.4f} seconds".format(
    correlation_time
))
print("Minimum correlation:", float(base_np.min()))
print("Maximum correlation:", float(base_np.max()))
print("Mean correlation:", float(base_np.mean()))


# ============================================================
# 7. BUILD MULTI-LEVEL CORRELATION PYRAMID
# ============================================================

print("\nBuilding correlation pyramid...")

# For each source location, treat its target-image correlation
# map as a separate 2D image.

batch, source_h, source_w, target_h, target_w = (
    correlation_volume.shape
)

correlation_maps = correlation_volume.reshape(
    batch * source_h * source_w,
    1,
    target_h,
    target_w
)

pyramid = [correlation_volume]

current_maps = correlation_maps

for level in range(1, PYRAMID_LEVELS):

    if current_maps.shape[-2] < 2 or current_maps.shape[-1] < 2:
        break

    current_maps = F.avg_pool2d(
        current_maps,
        kernel_size=2,
        stride=2
    )

    new_h, new_w = current_maps.shape[-2:]

    level_volume = current_maps.reshape(
        batch,
        source_h,
        source_w,
        new_h,
        new_w
    )

    pyramid.append(level_volume)

print("\nCORRELATION PYRAMID DIMENSIONS")
print("-" * 72)

for level, volume in enumerate(pyramid):

    print(
        "Level {}: {}".format(
            level,
            tuple(volume.shape)
        )
    )


# ============================================================
# 8. VISUALIZE CORRELATION MATRIX
# ============================================================

print("\nGenerating base correlation visualization...")

plt.figure(figsize=(11, 9))

plt.imshow(
    base_np.reshape(number_of_locations, number_of_locations),
    cmap="viridis",
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="Feature Correlation")
plt.xlabel("Image 2 Feature Locations")
plt.ylabel("Image 1 Feature Locations")
plt.title("Task 4 - Base All-Pairs Correlation")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "base_correlation_matrix.png",
    dpi=200
)
plt.close()


# ============================================================
# 9. VISUALIZE CORRELATION PYRAMID
# ============================================================

print("Generating correlation pyramid visualization...")

# Select one source feature location near the image centre.

source_row = GRID_HEIGHT // 2
source_col = GRID_WIDTH // 2

fig, axes = plt.subplots(
    1,
    len(pyramid),
    figsize=(5 * len(pyramid), 5)
)

if len(pyramid) == 1:
    axes = [axes]

for level, volume in enumerate(pyramid):

    correlation_map = (
        volume[0, source_row, source_col]
        .detach()
        .cpu()
        .numpy()
    )

    axes[level].imshow(
        correlation_map,
        cmap="viridis",
        interpolation="nearest"
    )

    axes[level].set_title(
        "Pyramid Level {}\n{} x {}".format(
            level,
            correlation_map.shape[0],
            correlation_map.shape[1]
        )
    )

    axes[level].set_xlabel("Target Feature X")
    axes[level].set_ylabel("Target Feature Y")

fig.suptitle(
    "Correlation Pyramid for One Source Feature Location",
    fontsize=15
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "correlation_pyramid_visualization.png",
    dpi=200
)
plt.close()


# ============================================================
# 10. VISUALIZE MULTIPLE SOURCE LOCATIONS
# ============================================================

print("Generating selected-location correlation maps...")

selected_locations = [
    (GRID_HEIGHT // 4, GRID_WIDTH // 4),
    (GRID_HEIGHT // 2, GRID_WIDTH // 2),
    (3 * GRID_HEIGHT // 4, 3 * GRID_WIDTH // 4)
]

fig, axes = plt.subplots(
    len(selected_locations),
    len(pyramid),
    figsize=(4 * len(pyramid), 4 * len(selected_locations))
)

for row_index, (source_y, source_x) in enumerate(selected_locations):

    for level, volume in enumerate(pyramid):

        correlation_map = (
            volume[0, source_y, source_x]
            .detach()
            .cpu()
            .numpy()
        )

        axes[row_index, level].imshow(
            correlation_map,
            cmap="viridis",
            interpolation="nearest"
        )

        axes[row_index, level].set_title(
            "Source ({}, {}) - Level {}".format(
                source_y, source_x, level
            )
        )

        axes[row_index, level].set_xlabel("Target X")
        axes[row_index, level].set_ylabel("Target Y")

fig.suptitle(
    "Correlation Maps for Selected Source Locations",
    fontsize=15
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "selected_location_correlations.png",
    dpi=200
)
plt.close()


# ============================================================
# 11. SAVE CORRELATION PYRAMID
# ============================================================

print("\nSaving correlation volume data...")

# Save all levels in one compressed file.
np.savez_compressed(
    OUTPUT_DIR / "correlation_volume_pyramid.npz",
    **{
        "level_{}".format(index):
            volume.squeeze(0).cpu().numpy()
        for index, volume in enumerate(pyramid)
    }
)

# Also save the base volume separately for convenient access.
np.save(
    OUTPUT_DIR / "base_correlation_volume.npy",
    base_np
)


# ============================================================
# 12. SAVE NUMERICAL SUMMARY
# ============================================================

summary_path = OUTPUT_DIR / "correlation_volume_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 4 - TASK 4: ALL-PAIRS CORRELATION VOLUME\n")
    file.write("=" * 65 + "\n\n")

    file.write("INPUT INFORMATION\n")
    file.write("-" * 40 + "\n")
    file.write("Image dimensions: {} x {}\n".format(width, height))
    file.write("Feature grid: {} x {}\n".format(
        GRID_HEIGHT, GRID_WIDTH
    ))
    file.write("Feature channels: 64\n")
    file.write("Feature locations: {}\n".format(number_of_locations))
    file.write("CNN parameters: {}\n".format(parameter_count))
    file.write("Random seed: {}\n".format(RANDOM_SEED))

    file.write("\nBASE CORRELATION VOLUME\n")
    file.write("-" * 40 + "\n")
    file.write("Shape: {}\n".format(tuple(base_np.shape)))
    file.write("Minimum: {:.8f}\n".format(float(base_np.min())))
    file.write("Maximum: {:.8f}\n".format(float(base_np.max())))
    file.write("Mean: {:.8f}\n".format(float(base_np.mean())))
    file.write("Standard deviation: {:.8f}\n".format(
        float(base_np.std())
    ))

    file.write("\nCORRELATION PYRAMID\n")
    file.write("-" * 40 + "\n")

    for level, volume in enumerate(pyramid):
        file.write(
            "Level {}: {}\n".format(
                level, tuple(volume.shape)
            )
        )

    file.write("\nPROCESSING TIME\n")
    file.write("-" * 40 + "\n")
    file.write("Feature extraction: {:.4f} seconds\n".format(
        feature_time
    ))
    file.write("Correlation computation: {:.4f} seconds\n".format(
        correlation_time
    ))

    file.write("\nIMPORTANT LIMITATION\n")
    file.write(
        "The CNN uses randomly initialized weights. "
        "This is an educational correlation-volume implementation, "
        "not a trained optical-flow model. The correlation pyramid "
        "demonstrates multi-scale matching information but does not "
        "produce optical flow by itself.\n"
    )


# ============================================================
# 13. COMPLETION
# ============================================================

print("\n" + "=" * 72)
print("STAGE 4 - TASK 4 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("Results saved to:", OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)
