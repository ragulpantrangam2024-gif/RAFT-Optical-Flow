
"""
STAGE 4 - TASK 5: CORRELATION LOOKUP AND LOCAL MATCHING

Demonstrates RAFT-style local correlation lookup using a
multi-level all-pairs correlation pyramid.

This is an educational implementation using a randomly
initialized CNN, not a trained RAFT model.
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

OUTPUT_DIR = PROJECT_DIR / "Stage_4" / "results" / "Task 5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_HEIGHT = 24
GRID_WIDTH = 40

PYRAMID_LEVELS = 4
LOOKUP_RADIUS = 2

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# 2. CNN FEATURE ENCODER
# Same architecture as Tasks 3 and 4
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

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool2d(2, 2)

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
# 3. IMAGE PREPROCESSING
# ============================================================

print("=" * 72)
print("STAGE 4 - TASK 5: CORRELATION LOOKUP AND LOCAL MATCHING")
print("=" * 72)

print("\nLoading KITTI image pair...")

image1 = cv2.imread(str(IMAGE1_PATH))
image2 = cv2.imread(str(IMAGE2_PATH))

if image1 is None or image2 is None:
    raise FileNotFoundError(
        "Could not load KITTI images. Check the dataset paths."
    )

if image1.shape != image2.shape:
    raise ValueError("Both images must have identical dimensions.")

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

print("Feature 1 shape:", tuple(features1.shape))
print("Feature 2 shape:", tuple(features2.shape))
print("Feature extraction time: {:.4f} seconds".format(feature_time))


# ============================================================
# 5. BUILD ALL-PAIRS CORRELATION VOLUME
# ============================================================

print("\nComputing all-pairs correlation...")

with torch.no_grad():

    feature1 = features1.flatten(2).transpose(1, 2)
    feature2 = features2.flatten(2).transpose(1, 2)

    feature1 = F.normalize(feature1, p=2, dim=-1)
    feature2 = F.normalize(feature2, p=2, dim=-1)

    correlation = torch.matmul(
        feature1,
        feature2.transpose(1, 2)
    )

    correlation_volume = correlation.reshape(
        1,
        GRID_HEIGHT,
        GRID_WIDTH,
        GRID_HEIGHT,
        GRID_WIDTH
    )

print("Base correlation shape:", tuple(correlation_volume.shape))


# ============================================================
# 6. BUILD CORRELATION PYRAMID
# ============================================================

print("\nBuilding correlation pyramid...")

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

    current_maps = F.avg_pool2d(
        current_maps,
        kernel_size=2,
        stride=2
    )

    level_h, level_w = current_maps.shape[-2:]

    level_volume = current_maps.reshape(
        batch,
        source_h,
        source_w,
        level_h,
        level_w
    )

    pyramid.append(level_volume)

for level, volume in enumerate(pyramid):
    print("Level {}: {}".format(level, tuple(volume.shape)))


# ============================================================
# 7. CREATE SOURCE COORDINATE GRID
# ============================================================

print("\nCreating source coordinate grid...")

# Coordinates are expressed in feature-grid pixels.
# x represents horizontal position.
# y represents vertical position.

y_coords, x_coords = torch.meshgrid(
    torch.arange(GRID_HEIGHT, device=DEVICE),
    torch.arange(GRID_WIDTH, device=DEVICE),
    indexing="ij"
)

coords = torch.stack(
    [x_coords, y_coords],
    dim=-1
).float()

coords = coords.unsqueeze(0)

# Initial flow estimate is zero.
# Therefore, candidate target positions initially equal
# source feature positions.

candidate_coords = coords.clone()

print("Coordinate grid shape:", tuple(coords.shape))
print("Initial displacement: (0, 0)")


# ============================================================
# 8. LOCAL CORRELATION LOOKUP FUNCTION
# ============================================================

def correlation_lookup(
    correlation_pyramid,
    candidate_coordinates,
    radius
):
    """
    Sample a local correlation neighbourhood at every
    source feature location and every pyramid level.

    Returns:
        lookup: [B, H, W, levels * neighbourhood_area]
    """

    batch_size, source_height, source_width, _ = (
        candidate_coordinates.shape
    )

    device = candidate_coordinates.device

    # Create local offsets:
    # For radius 2, offsets range from -2 to +2.
    offset_values = torch.arange(
        -radius,
        radius + 1,
        device=device
    )

    delta_y, delta_x = torch.meshgrid(
        offset_values,
        offset_values,
        indexing="ij"
    )

    offsets = torch.stack(
        [delta_x, delta_y],
        dim=-1
    ).float()

    offsets = offsets.reshape(-1, 2)

    neighbourhood_size = offsets.shape[0]

    sampled_levels = []

    for level, volume in enumerate(correlation_pyramid):

        target_height = volume.shape[-2]
        target_width = volume.shape[-1]

        # Each source feature location has one target
        # correlation map.
        maps = volume.reshape(
            batch_size * source_height * source_width,
            1,
            target_height,
            target_width
        )

        # Convert candidate coordinates to this pyramid level.
        level_coordinates = (
            candidate_coordinates / (2 ** level)
        )

        level_coordinates = level_coordinates.reshape(
            batch_size * source_height * source_width,
            1,
            1,
            2
        )

        # Add local offsets around each candidate location.
        sample_coordinates = (
            level_coordinates + offsets.reshape(
                1, 1, neighbourhood_size, 2
            )
        )

        # Normalize coordinates for grid_sample.
        # align_corners=True maps pixel centres to [-1, 1].
        if target_width > 1:
            normalized_x = (
                2.0 * sample_coordinates[..., 0]
                / (target_width - 1)
            ) - 1.0
        else:
            normalized_x = torch.zeros_like(
                sample_coordinates[..., 0]
            )

        if target_height > 1:
            normalized_y = (
                2.0 * sample_coordinates[..., 1]
                / (target_height - 1)
            ) - 1.0
        else:
            normalized_y = torch.zeros_like(
                sample_coordinates[..., 1]
            )

        sampling_grid = torch.stack(
            [normalized_x, normalized_y],
            dim=-1
        )

        # Sample all local correlation values.
        sampled = F.grid_sample(
            maps,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        # [B*H*W, 1, 1, neighbourhood_size]
        sampled = sampled.reshape(
            batch_size,
            source_height,
            source_width,
            neighbourhood_size
        )

        sampled_levels.append(sampled)

    # Concatenate all pyramid-level neighbourhoods.
    lookup = torch.cat(sampled_levels, dim=-1)

    return lookup


# ============================================================
# 9. PERFORM LOCAL CORRELATION LOOKUP
# ============================================================

print("\nPerforming local correlation lookup...")

start_time = time.perf_counter()

with torch.no_grad():

    lookup_features = correlation_lookup(
        pyramid,
        candidate_coords,
        LOOKUP_RADIUS
    )

lookup_time = time.perf_counter() - start_time

neighbourhood_area = (
    (2 * LOOKUP_RADIUS + 1) ** 2
)

expected_channels = (
    PYRAMID_LEVELS * neighbourhood_area
)

print("Lookup output shape:", tuple(lookup_features.shape))
print("Lookup radius:", LOOKUP_RADIUS)
print("Neighbourhood positions per level:", neighbourhood_area)
print("Number of pyramid levels:", len(pyramid))
print("Expected lookup channels:", expected_channels)
print("Actual lookup channels:", lookup_features.shape[-1])
print("Lookup time: {:.4f} seconds".format(lookup_time))

if lookup_features.shape[-1] != expected_channels:
    raise RuntimeError("Unexpected correlation lookup dimensions.")


# ============================================================
# 10. VISUALIZE ONE SOURCE LOCATION
# ============================================================

print("\nGenerating local lookup visualizations...")

source_y = GRID_HEIGHT // 2
source_x = GRID_WIDTH // 2

fig, axes = plt.subplots(
    1,
    PYRAMID_LEVELS,
    figsize=(5 * PYRAMID_LEVELS, 5)
)

for level in range(PYRAMID_LEVELS):

    start = level * neighbourhood_area
    end = start + neighbourhood_area

    local_values = (
        lookup_features[
            0, source_y, source_x, start:end
        ]
        .detach()
        .cpu()
        .numpy()
    )

    local_map = local_values.reshape(
        2 * LOOKUP_RADIUS + 1,
        2 * LOOKUP_RADIUS + 1
    )

    image_plot = axes[level].imshow(
        local_map,
        cmap="viridis",
        interpolation="nearest"
    )

    axes[level].set_title(
        "Level {}\nLocal correlation".format(level)
    )

    axes[level].set_xlabel("Neighbourhood X")
    axes[level].set_ylabel("Neighbourhood Y")

    axes[level].set_xticks(
        range(2 * LOOKUP_RADIUS + 1)
    )
    axes[level].set_yticks(
        range(2 * LOOKUP_RADIUS + 1)
    )

    axes[level].scatter(
        [LOOKUP_RADIUS],
        [LOOKUP_RADIUS],
        marker="x",
        color="red",
        s=100,
        label="Candidate centre"
    )

    axes[level].legend(loc="lower right")
    fig.colorbar(image_plot, ax=axes[level], fraction=0.046)

fig.suptitle(
    "Local Correlation Lookup Around One Source Feature",
    fontsize=15
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "local_correlation_lookup.png",
    dpi=200
)
plt.close()


# ============================================================
# 11. VISUALIZE LOOKUP RESPONSE ACROSS SOURCE GRID
# ============================================================

print("Generating lookup response maps...")

fig, axes = plt.subplots(
    1,
    PYRAMID_LEVELS,
    figsize=(5 * PYRAMID_LEVELS, 5)
)

for level in range(PYRAMID_LEVELS):

    start = level * neighbourhood_area
    end = start + neighbourhood_area

    level_lookup = lookup_features[
        ..., start:end
    ]

    # Maximum correlation in the local neighbourhood.
    maximum_response = level_lookup.max(dim=-1).values

    response_np = (
        maximum_response[0].detach().cpu().numpy()
    )

    image_plot = axes[level].imshow(
        response_np,
        cmap="viridis",
        interpolation="nearest"
    )

    axes[level].set_title(
        "Level {} - Maximum Local Correlation".format(level)
    )

    axes[level].set_xlabel("Source X")
    axes[level].set_ylabel("Source Y")

    fig.colorbar(image_plot, ax=axes[level], fraction=0.046)

fig.suptitle(
    "Maximum Local Correlation Across Source Feature Grid",
    fontsize=15
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "lookup_response_maps.png",
    dpi=200
)
plt.close()


# ============================================================
# 12. SAVE LOOKUP DATA
# ============================================================

print("\nSaving correlation lookup data...")

lookup_np = lookup_features.squeeze(0).cpu().numpy()

np.save(
    OUTPUT_DIR / "correlation_lookup_features.npy",
    lookup_np
)

np.save(
    OUTPUT_DIR / "candidate_coordinates.npy",
    candidate_coords.squeeze(0).cpu().numpy()
)

# Save the feature pyramid for reproducibility of this run.
np.savez_compressed(
    OUTPUT_DIR / "correlation_lookup_pyramid.npz",
    **{
        "level_{}".format(index):
            volume.squeeze(0).cpu().numpy()
        for index, volume in enumerate(pyramid)
    }
)


# ============================================================
# 13. NUMERICAL SUMMARY
# ============================================================

lookup_min = float(lookup_np.min())
lookup_max = float(lookup_np.max())
lookup_mean = float(lookup_np.mean())
lookup_std = float(lookup_np.std())

summary_path = OUTPUT_DIR / "correlation_lookup_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 4 - TASK 5: CORRELATION LOOKUP\n")
    file.write("=" * 65 + "\n\n")

    file.write("INPUT INFORMATION\n")
    file.write("-" * 40 + "\n")
    file.write("Image dimensions: {} x {}\n".format(width, height))
    file.write("Feature grid: {} x {}\n".format(
        GRID_HEIGHT, GRID_WIDTH
    ))
    file.write("Feature channels: 64\n")
    file.write("Pyramid levels: {}\n".format(PYRAMID_LEVELS))
    file.write("Lookup radius: {}\n".format(LOOKUP_RADIUS))
    file.write("Random seed: {}\n".format(RANDOM_SEED))

    file.write("\nLOOKUP DIMENSIONS\n")
    file.write("-" * 40 + "\n")
    file.write("Source locations: {}\n".format(
        GRID_HEIGHT * GRID_WIDTH
    ))
    file.write("Neighbourhood positions per level: {}\n".format(
        neighbourhood_area
    ))
    file.write("Total lookup channels: {}\n".format(
        expected_channels
    ))
    file.write("Lookup tensor shape: {}\n".format(
        tuple(lookup_features.shape)
    ))

    file.write("\nLOOKUP STATISTICS\n")
    file.write("-" * 40 + "\n")
    file.write("Minimum: {:.8f}\n".format(lookup_min))
    file.write("Maximum: {:.8f}\n".format(lookup_max))
    file.write("Mean: {:.8f}\n".format(lookup_mean))
    file.write("Standard deviation: {:.8f}\n".format(lookup_std))

    file.write("\nPROCESSING TIME\n")
    file.write("-" * 40 + "\n")
    file.write("Feature extraction: {:.4f} seconds\n".format(
        feature_time
    ))
    file.write("Local correlation lookup: {:.4f} seconds\n".format(
        lookup_time
    ))

    file.write("\nIMPORTANT LIMITATION\n")
    file.write(
        "The CNN uses randomly initialized weights. "
        "The initial candidate flow is zero. This task demonstrates "
        "multi-scale local correlation lookup, not trained optical "
        "flow estimation or RAFT iterative updates.\n"
    )


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 72)
print("STAGE 4 - TASK 5 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("Lookup tensor shape:", tuple(lookup_features.shape))
print("Lookup channels:", lookup_features.shape[-1])
print("Lookup mean:", lookup_mean)
print("Lookup time: {:.4f} seconds".format(lookup_time))

print("\nResults saved to:", OUTPUT_DIR)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)
