
"""
STAGE 1 - TASK 4
DENSE OPTICAL FLOW USING FARNEBACK

Dataset: MPI Sintel - Clean
Sequence: alley_1
Frames: frame_0001.png and frame_0002.png

This script:
1. Loads two consecutive grayscale images.
2. Calculates dense optical flow using Farneback.
3. Extracts horizontal and vertical flow components.
4. Calculates motion magnitude and direction.
5. Creates HSV optical flow visualization.
6. Creates motion magnitude visualization.
7. Saves numerical flow data and statistics.
"""

from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "MPI-Sintel-training_images"
    / "training"
    / "clean"
    / "alley_1"
)

FRAME_1_PATH = DATASET_DIR / "frame_0001.png"
FRAME_2_PATH = DATASET_DIR / "frame_0002.png"

RESULTS_DIR = PROJECT_ROOT / "Stage_1" / "results" / "Task 4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD INPUT IMAGES
# --------------------------------------------------

frame1 = cv2.imread(str(FRAME_1_PATH), cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(str(FRAME_2_PATH), cv2.IMREAD_GRAYSCALE)

if frame1 is None or frame2 is None:
    raise FileNotFoundError(
        "Could not load input images. Check the dataset paths."
    )

if frame1.shape != frame2.shape:
    raise ValueError("Both input images must have the same dimensions.")

height, width = frame1.shape

print("=" * 60)
print("STAGE 1 - TASK 4")
print("DENSE OPTICAL FLOW USING FARNEBACK")
print("=" * 60)

print(f"\nImage dimensions: {width} x {height}")


# --------------------------------------------------
# 3. FARNEBACK PARAMETERS
# --------------------------------------------------

farneback_params = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0
}


# --------------------------------------------------
# 4. CALCULATE DENSE OPTICAL FLOW
# --------------------------------------------------

flow = cv2.calcOpticalFlowFarneback(
    frame1,
    frame2,
    None,
    **farneback_params
)

# Flow has two components at every pixel:
# flow[..., 0] = horizontal displacement
# flow[..., 1] = vertical displacement

u = flow[..., 0]
v = flow[..., 1]

print("\nDense optical flow calculated successfully.")


# --------------------------------------------------
# 5. CALCULATE MAGNITUDE AND DIRECTION
# --------------------------------------------------

magnitude, angle = cv2.cartToPolar(
    u,
    v,
    angleInDegrees=True
)

print("\nFLOW FIELD STATISTICS")
print("-" * 45)

print(f"Total pixels: {width * height}")
print(f"Flow array shape: {flow.shape}")

print(f"\nHorizontal flow (u):")
print(f"  Minimum: {np.min(u):.6f} pixels")
print(f"  Maximum: {np.max(u):.6f} pixels")
print(f"  Mean:    {np.mean(u):.6f} pixels")
print(f"  Std Dev: {np.std(u):.6f} pixels")

print(f"\nVertical flow (v):")
print(f"  Minimum: {np.min(v):.6f} pixels")
print(f"  Maximum: {np.max(v):.6f} pixels")
print(f"  Mean:    {np.mean(v):.6f} pixels")
print(f"  Std Dev: {np.std(v):.6f} pixels")

print(f"\nMotion magnitude:")
print(f"  Minimum: {np.min(magnitude):.6f} pixels")
print(f"  Maximum: {np.max(magnitude):.6f} pixels")
print(f"  Mean:    {np.mean(magnitude):.6f} pixels")
print(f"  Median:  {np.median(magnitude):.6f} pixels")


# --------------------------------------------------
# 6. HSV COLOR-CODED OPTICAL FLOW
# --------------------------------------------------

# Hue represents motion direction.
# Saturation is kept at maximum.
# Value represents normalized motion magnitude.

hsv = np.zeros((height, width, 3), dtype=np.uint8)

# OpenCV HSV hue range is 0 to 179.
hsv[..., 0] = np.mod(angle / 2, 180).astype(np.uint8)
hsv[..., 1] = 255

# Use the 99th percentile to avoid a few large
# vectors dominating the visualization contrast.
magnitude_scale = np.percentile(magnitude, 99)

if magnitude_scale > 0:
    normalized_magnitude = np.clip(
        magnitude / magnitude_scale,
        0,
        1
    )
else:
    normalized_magnitude = np.zeros_like(magnitude)

hsv[..., 2] = (normalized_magnitude * 255).astype(np.uint8)

flow_color = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2BGR
)

cv2.imwrite(
    str(RESULTS_DIR / "farneback_dense_optical_flow.png"),
    flow_color
)


# --------------------------------------------------
# 7. MOTION MAGNITUDE VISUALIZATION
# --------------------------------------------------

magnitude_visualization = (
    normalized_magnitude * 255
).astype(np.uint8)

magnitude_color = cv2.applyColorMap(
    magnitude_visualization,
    cv2.COLORMAP_JET
)

cv2.imwrite(
    str(RESULTS_DIR / "motion_magnitude.png"),
    magnitude_color
)


# --------------------------------------------------
# 8. MOTION VECTOR QUIVER VISUALIZATION
# --------------------------------------------------

# Draw a vector at regularly spaced pixel locations.
# This makes the dense flow field easier to inspect.

vector_image = cv2.cvtColor(
    frame1,
    cv2.COLOR_GRAY2BGR
)

step = 20

for y in range(0, height, step):
    for x in range(0, width, step):

        dx = float(u[y, x])
        dy = float(v[y, x])

        # Skip extremely small vectors for clarity.
        if np.hypot(dx, dy) < 0.5:
            continue

        start_point = (x, y)

        end_point = (
            int(round(x + dx)),
            int(round(y + dy))
        )

        cv2.arrowedLine(
            vector_image,
            start_point,
            end_point,
            (0, 255, 0),
            1,
            tipLength=0.3
        )

cv2.imwrite(
    str(RESULTS_DIR / "dense_flow_vectors.png"),
    vector_image
)


# --------------------------------------------------
# 9. SAVE NUMERICAL FLOW DATA
# --------------------------------------------------

# Shape: height x width x 2
# Channel 0: horizontal flow u
# Channel 1: vertical flow v

np.save(
    RESULTS_DIR / "farneback_flow.npy",
    flow
)

np.save(
    RESULTS_DIR / "flow_magnitude.npy",
    magnitude
)

np.save(
    RESULTS_DIR / "flow_direction.npy",
    angle
)


# --------------------------------------------------
# 10. SAVE FLOW STATISTICS
# --------------------------------------------------

summary = f"""
DENSE OPTICAL FLOW - TASK 4

Dataset: MPI Sintel
Sequence: alley_1
Frames: frame_0001.png -> frame_0002.png

Image dimensions: {width} x {height}
Total pixels: {width * height}

Algorithm: Farneback Dense Optical Flow

Parameters:
    Pyramid scale: {farneback_params['pyr_scale']}
    Pyramid levels: {farneback_params['levels']}
    Window size: {farneback_params['winsize']}
    Iterations: {farneback_params['iterations']}
    Polynomial neighbourhood: {farneback_params['poly_n']}
    Polynomial sigma: {farneback_params['poly_sigma']}

Horizontal flow (u):
    Minimum: {np.min(u):.6f}
    Maximum: {np.max(u):.6f}
    Mean: {np.mean(u):.6f}
    Standard deviation: {np.std(u):.6f}

Vertical flow (v):
    Minimum: {np.min(v):.6f}
    Maximum: {np.max(v):.6f}
    Mean: {np.mean(v):.6f}
    Standard deviation: {np.std(v):.6f}

Motion magnitude:
    Minimum: {np.min(magnitude):.6f}
    Maximum: {np.max(magnitude):.6f}
    Mean: {np.mean(magnitude):.6f}
    Median: {np.median(magnitude):.6f}

Note:
    This is dense optical flow.
    Motion is estimated for every pixel.
    The color visualization uses normalized magnitude.
"""

(RESULTS_DIR / "flow_summary.txt").write_text(
    summary,
    encoding="utf-8"
)


# --------------------------------------------------
# 11. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("TASK 4 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nResults saved in:\n{RESULTS_DIR}")

print("\nGenerated files:")
print("  - farneback_dense_optical_flow.png")
print("  - motion_magnitude.png")
print("  - dense_flow_vectors.png")
print("  - farneback_flow.npy")
print("  - flow_magnitude.npy")
print("  - flow_direction.npy")
print("  - flow_summary.txt")
