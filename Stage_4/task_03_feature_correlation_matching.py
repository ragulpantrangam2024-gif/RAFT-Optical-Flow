
"""
STAGE 4 - TASK 3: CNN FEATURE CORRELATION AND MATCHING

Purpose:
1. Extract CNN features from two KITTI frames.
2. Normalize feature vectors.
3. Compute an all-pairs correlation matrix.
4. Find mutual nearest-neighbour feature matches.
5. Visualize correlation and matched locations.

Note:
The CNN is randomly initialized. This is an educational
correlation experiment, not a trained optical-flow model.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
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

OUTPUT_DIR = PROJECT_DIR / "Stage_4" / "results" / "Task 3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reduced spatial grid for manageable all-pairs correlation.
GRID_HEIGHT = 24
GRID_WIDTH = 40

# Maximum number of matches displayed.
MAX_DISPLAY_MATCHES = 120


# ============================================================
# 2. CNN FEATURE ENCODER
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
# 3. LOAD AND PREPROCESS IMAGES
# ============================================================

print("=" * 70)
print("STAGE 4 - TASK 3: FEATURE CORRELATION AND MATCHING")
print("=" * 70)

print("\nLoading KITTI images...")

image1 = cv2.imread(str(IMAGE1_PATH))
image2 = cv2.imread(str(IMAGE2_PATH))

if image1 is None or image2 is None:
    raise FileNotFoundError(
        "Could not load KITTI images. Check the image paths."
    )

if image1.shape != image2.shape:
    raise ValueError("The two input images must have identical dimensions.")

height, width = image1.shape[:2]

print("Image shape:", image1.shape)
print("Processing device:", DEVICE)

def preprocess_image(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(rgb.copy())
    tensor = tensor.permute(2, 0, 1).float() / 255.0

    return tensor.unsqueeze(0)


tensor1 = preprocess_image(image1).to(DEVICE)
tensor2 = preprocess_image(image2).to(DEVICE)


# ============================================================
# 4. EXTRACT CNN FEATURES
# ============================================================

print("\nInitializing shared CNN encoder...")

model = CNNFeatureEncoder().to(DEVICE)
model.eval()

parameter_count = sum(p.numel() for p in model.parameters())

print("Total CNN parameters:", parameter_count)

print("\nExtracting features...")

start_time = time.perf_counter()

with torch.no_grad():

    features1 = model(tensor1)
    features2 = model(tensor2)

    # Reduce feature-map resolution.
    features1 = F.adaptive_avg_pool2d(
        features1, (GRID_HEIGHT, GRID_WIDTH)
    )

    features2 = F.adaptive_avg_pool2d(
        features2, (GRID_HEIGHT, GRID_WIDTH)
    )

elapsed_time = time.perf_counter() - start_time

print("Feature extraction time: {:.4f} seconds".format(elapsed_time))
print("Feature 1 shape:", tuple(features1.shape))
print("Feature 2 shape:", tuple(features2.shape))


# ============================================================
# 5. NORMALIZE AND FLATTEN FEATURES
# ============================================================

print("\nNormalizing feature vectors...")

# [1, 64, 24, 40] -> [960, 64]
feature1 = features1.squeeze(0).permute(1, 2, 0)
feature2 = features2.squeeze(0).permute(1, 2, 0)

feature1 = feature1.reshape(-1, 64)
feature2 = feature2.reshape(-1, 64)

# L2 normalization makes dot product equal cosine similarity.
feature1 = F.normalize(feature1, p=2, dim=1)
feature2 = F.normalize(feature2, p=2, dim=1)

number_of_locations = feature1.shape[0]

print("Feature locations per image:", number_of_locations)
print("Feature dimension:", feature1.shape[1])


# ============================================================
# 6. COMPUTE ALL-PAIRS CORRELATION
# ============================================================

print("\nComputing all-pairs correlation...")

with torch.no_grad():

    correlation = torch.matmul(
        feature1,
        feature2.transpose(0, 1)
    )

correlation_np = correlation.cpu().numpy()

print("Correlation matrix shape:", correlation_np.shape)
print("Correlation minimum:", float(correlation_np.min()))
print("Correlation maximum:", float(correlation_np.max()))
print("Correlation mean:", float(correlation_np.mean()))


# ============================================================
# 7. MUTUAL NEAREST-NEIGHBOUR MATCHING
# ============================================================

print("\nFinding mutual nearest-neighbour matches...")

# Best match in Image 2 for each Image 1 location.
best_match_1_to_2 = np.argmax(correlation_np, axis=1)

# Best match in Image 1 for each Image 2 location.
best_match_2_to_1 = np.argmax(correlation_np, axis=0)

matches = []

for index1, index2 in enumerate(best_match_1_to_2):

    # Keep only mutually consistent matches.
    if best_match_2_to_1[index2] == index1:

        similarity = correlation_np[index1, index2]

        matches.append(
            (index1, index2, float(similarity))
        )

matches.sort(key=lambda item: item[2], reverse=True)

print("Mutual matches found:", len(matches))

if matches:
    similarities = np.array([m[2] for m in matches])

    print("Mean match similarity:", float(similarities.mean()))
    print("Maximum match similarity:", float(similarities.max()))
    print("Minimum match similarity:", float(similarities.min()))


# ============================================================
# 8. SAVE CORRELATION MATRIX
# ============================================================

print("\nGenerating correlation visualization...")

plt.figure(figsize=(10, 8))

plt.imshow(
    correlation_np,
    cmap="viridis",
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="Cosine Similarity")
plt.xlabel("Image 2 Feature Locations")
plt.ylabel("Image 1 Feature Locations")
plt.title("All-Pairs CNN Feature Correlation")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "all_pairs_correlation.png",
    dpi=200
)
plt.close()

np.save(
    OUTPUT_DIR / "correlation_matrix.npy",
    correlation_np
)


# ============================================================
# 9. VISUALIZE MATCHED LOCATIONS
# ============================================================

print("\nVisualizing feature matches...")

rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

canvas = np.zeros(
    (height, width * 2, 3),
    dtype=np.uint8
)

canvas[:, :width] = rgb1
canvas[:, width:] = rgb2

display_matches = matches[:MAX_DISPLAY_MATCHES]

for index1, index2, similarity in display_matches:

    row1, col1 = divmod(index1, GRID_WIDTH)
    row2, col2 = divmod(index2, GRID_WIDTH)

    x1 = int((col1 + 0.5) * width / GRID_WIDTH)
    y1 = int((row1 + 0.5) * height / GRID_HEIGHT)

    x2 = int((col2 + 0.5) * width / GRID_WIDTH) + width
    y2 = int((row2 + 0.5) * height / GRID_HEIGHT)

    cv2.circle(canvas, (x1, y1), 3, (0, 255, 0), -1)
    cv2.circle(canvas, (x2, y2), 3, (0, 255, 0), -1)

    cv2.line(
        canvas,
        (x1, y1),
        (x2, y2),
        (255, 80, 0),
        1
    )

plt.figure(figsize=(18, 6))
plt.imshow(canvas)
plt.title("Mutual Nearest-Neighbor CNN Feature Matches")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_matches.png",
    dpi=200
)
plt.close()


# ============================================================
# 10. VISUALIZE MATCH DISPLACEMENTS
# ============================================================

print("\nGenerating displacement visualization...")

plt.figure(figsize=(14, 6))
plt.imshow(rgb1)

if display_matches:

    xs = []
    ys = []
    us = []
    vs = []

    for index1, index2, similarity in display_matches:

        row1, col1 = divmod(index1, GRID_WIDTH)
        row2, col2 = divmod(index2, GRID_WIDTH)

        x1 = (col1 + 0.5) * width / GRID_WIDTH
        y1 = (row1 + 0.5) * height / GRID_HEIGHT

        x2 = (col2 + 0.5) * width / GRID_WIDTH
        y2 = (row2 + 0.5) * height / GRID_HEIGHT

        xs.append(x1)
        ys.append(y1)
        us.append(x2 - x1)
        vs.append(y2 - y1)

    plt.quiver(
        xs, ys, us, vs,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.002
    )

plt.title("Displacement of Matched CNN Feature Locations")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_displacements.png",
    dpi=200
)
plt.close()


# ============================================================
# 11. SAVE MATCH METRICS
# ============================================================

print("\nSaving numerical results...")

match_rows = []

for rank, (index1, index2, similarity) in enumerate(matches, start=1):

    row1, col1 = divmod(index1, GRID_WIDTH)
    row2, col2 = divmod(index2, GRID_WIDTH)

    x1 = (col1 + 0.5) * width / GRID_WIDTH
    y1 = (row1 + 0.5) * height / GRID_HEIGHT

    x2 = (col2 + 0.5) * width / GRID_WIDTH
    y2 = (row2 + 0.5) * height / GRID_HEIGHT

    match_rows.append({
        "rank": rank,
        "image1_x": x1,
        "image1_y": y1,
        "image2_x": x2,
        "image2_y": y2,
        "displacement_x": x2 - x1,
        "displacement_y": y2 - y1,
        "cosine_similarity": similarity
    })

df = pd.DataFrame(match_rows)

df.to_csv(
    OUTPUT_DIR / "feature_matches.csv",
    index=False
)

summary_path = OUTPUT_DIR / "feature_correlation_summary.txt"

with open(summary_path, "w", encoding="utf-8") as file:

    file.write("STAGE 4 - TASK 3: FEATURE CORRELATION AND MATCHING\n")
    file.write("=" * 65 + "\n\n")

    file.write(f"Image dimensions: {width} x {height}\n")
    file.write(f"Feature grid: {GRID_HEIGHT} x {GRID_WIDTH}\n")
    file.write(f"Feature dimension: 64\n")
    file.write(f"Feature locations per image: {number_of_locations}\n")
    file.write(
        f"Correlation matrix: {correlation_np.shape}\n"
    )

    file.write(f"\nCNN parameters: {parameter_count}\n")
    file.write(f"Feature extraction time: {elapsed_time:.4f} seconds\n")

    file.write(
        f"\nMutual nearest-neighbour matches: {len(matches)}\n"
    )

    file.write(
        f"Correlation minimum: {correlation_np.min():.6f}\n"
    )
    file.write(
        f"Correlation maximum: {correlation_np.max():.6f}\n"
    )
    file.write(
        f"Correlation mean: {correlation_np.mean():.6f}\n"
    )

    if matches:
        file.write(
            f"Mean match similarity: {similarities.mean():.6f}\n"
        )

    file.write("\nIMPORTANT LIMITATION\n")
    file.write(
        "The CNN uses randomly initialized weights. "
        "The matches are not verified geometric correspondences "
        "and should not be interpreted as accurate optical flow.\n"
    )

print("Saved feature_matches.csv")
print("Saved feature_correlation_summary.txt")


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("STAGE 4 - TASK 3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Results saved to:", OUTPUT_DIR)

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(" -", path.name)
