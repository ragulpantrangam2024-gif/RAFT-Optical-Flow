
"""
STAGE 2 - TASK 1
HORN-SCHUNCK DENSE OPTICAL FLOW

Dataset: MPI Sintel - Clean
Sequence: alley_1
Frames: frame_0001.png -> frame_0002.png

Implementation:
    Horn-Schunck global optical flow from scratch.

Outputs:
    - Dense optical flow
    - HSV flow visualization
    - Motion magnitude visualization
    - Flow vector visualization
    - Numerical flow arrays
    - Statistical summary
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

RESULTS_DIR = (
    PROJECT_ROOT
    / "Stage_2"
    / "results"
    / "Task 1"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. HORN-SCHUNCK PARAMETERS
# --------------------------------------------------

ALPHA = 1.0
ITERATIONS = 100

# Neighbour averaging kernel.
AVERAGING_KERNEL = np.array([
    [1 / 12, 1 / 6, 1 / 12],
    [1 / 6,  0,    1 / 6],
    [1 / 12, 1 / 6, 1 / 12]
], dtype=np.float32)


# --------------------------------------------------
# 3. LOAD IMAGES
# --------------------------------------------------

frame1 = cv2.imread(
    str(FRAME_1_PATH),
    cv2.IMREAD_GRAYSCALE
)

frame2 = cv2.imread(
    str(FRAME_2_PATH),
    cv2.IMREAD_GRAYSCALE
)

if frame1 is None or frame2 is None:
    raise FileNotFoundError(
        "Could not load Sintel input images. Check dataset paths."
    )

if frame1.shape != frame2.shape:
    raise ValueError("Input frames must have identical dimensions.")

height, width = frame1.shape

print("=" * 65)
print("STAGE 2 - TASK 1")
print("HORN-SCHUNCK DENSE OPTICAL FLOW")
print("=" * 65)

print(f"\nImage dimensions: {width} x {height}")
print(f"Regularization parameter (alpha): {ALPHA}")
print(f"Iterations: {ITERATIONS}")


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

# Normalize grayscale intensities to [0, 1].

I1 = frame1.astype(np.float32) / 255.0
I2 = frame2.astype(np.float32) / 255.0


# --------------------------------------------------
# 5. CALCULATE IMAGE DERIVATIVES
# --------------------------------------------------

# Average spatial derivatives from both frames.

Ix1 = cv2.Sobel(
    I1, cv2.CV_32F, 1, 0, ksize=3,
    scale=1 / 8.0
)

Ix2 = cv2.Sobel(
    I2, cv2.CV_32F, 1, 0, ksize=3,
    scale=1 / 8.0
)

Iy1 = cv2.Sobel(
    I1, cv2.CV_32F, 0, 1, ksize=3,
    scale=1 / 8.0
)

Iy2 = cv2.Sobel(
    I2, cv2.CV_32F, 0, 1, ksize=3,
    scale=1 / 8.0
)

Ix = (Ix1 + Ix2) / 2.0
Iy = (Iy1 + Iy2) / 2.0

# Temporal derivative.

It = I2 - I1

print("\nImage derivatives calculated successfully.")


# --------------------------------------------------
# 6. HORN-SCHUNCK ITERATIVE OPTIMIZATION
# --------------------------------------------------

def horn_schunck(Ix, Iy, It, alpha, iterations):

    """
    Estimate dense optical flow using Horn-Schunck.

    Parameters:
        Ix: Horizontal image derivative
        Iy: Vertical image derivative
        It: Temporal image derivative
        alpha: Smoothness regularization parameter
        iterations: Number of iterations

    Returns:
        u: Horizontal flow
        v: Vertical flow
    """

    height, width = Ix.shape

    # Initialize flow fields.

    u = np.zeros((height, width), dtype=np.float32)
    v = np.zeros((height, width), dtype=np.float32)

    # Precompute terms.

    Ix_squared = Ix ** 2
    Iy_squared = Iy ** 2

    denominator = alpha ** 2 + Ix_squared + Iy_squared

    # Iterative estimation.

    for iteration in range(iterations):

        # Estimate average flow from neighbouring pixels.

        u_average = cv2.filter2D(
            u,
            -1,
            AVERAGING_KERNEL,
            borderType=cv2.BORDER_REFLECT
        )

        v_average = cv2.filter2D(
            v,
            -1,
            AVERAGING_KERNEL,
            borderType=cv2.BORDER_REFLECT
        )

        # Brightness constancy residual.

        residual = (
            Ix * u_average
            + Iy * v_average
            + It
        )

        # Update horizontal and vertical flow.

        u = u_average - (
            Ix * residual / denominator
        )

        v = v_average - (
            Iy * residual / denominator
        )

        if (iteration + 1) % 20 == 0:
            print(
                f"Iteration {iteration + 1}/{iterations} completed"
            )

    return u, v


# --------------------------------------------------
# 7. RUN HORN-SCHUNCK
# --------------------------------------------------

u, v = horn_schunck(
    Ix,
    Iy,
    It,
    ALPHA,
    ITERATIONS
)

flow = np.dstack((u, v))

print("\nHorn-Schunck optical flow calculated successfully.")


# --------------------------------------------------
# 8. CALCULATE MAGNITUDE AND DIRECTION
# --------------------------------------------------

magnitude, angle = cv2.cartToPolar(
    u,
    v,
    angleInDegrees=True
)

print("\nFLOW STATISTICS")
print("-" * 50)

print("\nHorizontal flow (u):")
print(f"  Minimum: {np.min(u):.6f}")
print(f"  Maximum: {np.max(u):.6f}")
print(f"  Mean:    {np.mean(u):.6f}")
print(f"  Std Dev: {np.std(u):.6f}")

print("\nVertical flow (v):")
print(f"  Minimum: {np.min(v):.6f}")
print(f"  Maximum: {np.max(v):.6f}")
print(f"  Mean:    {np.mean(v):.6f}")
print(f"  Std Dev: {np.std(v):.6f}")

print("\nMotion magnitude:")
print(f"  Minimum: {np.min(magnitude):.6f}")
print(f"  Maximum: {np.max(magnitude):.6f}")
print(f"  Mean:    {np.mean(magnitude):.6f}")
print(f"  Median:  {np.median(magnitude):.6f}")


# --------------------------------------------------
# 9. HSV FLOW VISUALIZATION
# --------------------------------------------------

hsv = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

hsv[..., 0] = np.mod(
    angle / 2,
    180
).astype(np.uint8)

hsv[..., 1] = 255

magnitude_scale = np.percentile(
    magnitude,
    99
)

if magnitude_scale > 0:

    normalized_magnitude = np.clip(
        magnitude / magnitude_scale,
        0,
        1
    )

else:

    normalized_magnitude = np.zeros_like(magnitude)

hsv[..., 2] = (
    normalized_magnitude * 255
).astype(np.uint8)

flow_color = cv2.cvtColor(
    hsv,
    cv2.COLOR_HSV2BGR
)

cv2.imwrite(
    str(RESULTS_DIR / "horn_schunck_flow.png"),
    flow_color
)


# --------------------------------------------------
# 10. MOTION MAGNITUDE VISUALIZATION
# --------------------------------------------------

magnitude_image = (
    normalized_magnitude * 255
).astype(np.uint8)

magnitude_color = cv2.applyColorMap(
    magnitude_image,
    cv2.COLORMAP_JET
)

cv2.imwrite(
    str(RESULTS_DIR / "motion_magnitude.png"),
    magnitude_color
)


# --------------------------------------------------
# 11. FLOW VECTOR VISUALIZATION
# --------------------------------------------------

vector_image = cv2.cvtColor(
    frame1,
    cv2.COLOR_GRAY2BGR
)

step = 20

for y in range(0, height, step):

    for x in range(0, width, step):

        dx = float(u[y, x])
        dy = float(v[y, x])

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
    str(RESULTS_DIR / "horn_schunck_vectors.png"),
    vector_image
)


# --------------------------------------------------
# 12. SAVE NUMERICAL FLOW ARRAYS
# --------------------------------------------------

np.save(
    RESULTS_DIR / "horn_schunck_flow.npy",
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
# 13. SAVE SUMMARY
# --------------------------------------------------

summary = f"""
STAGE 2 - TASK 1
HORN-SCHUNCK DENSE OPTICAL FLOW

Dataset: MPI Sintel
Sequence: alley_1
Frames: frame_0001.png -> frame_0002.png

Image dimensions: {width} x {height}

Algorithm: Horn-Schunck
Implementation: Custom NumPy and OpenCV implementation

Parameters:
    Alpha: {ALPHA}
    Iterations: {ITERATIONS}

Horizontal flow:
    Minimum: {np.min(u):.6f}
    Maximum: {np.max(u):.6f}
    Mean: {np.mean(u):.6f}
    Standard deviation: {np.std(u):.6f}

Vertical flow:
    Minimum: {np.min(v):.6f}
    Maximum: {np.max(v):.6f}
    Mean: {np.mean(v):.6f}
    Standard deviation: {np.std(v):.6f}

Motion magnitude:
    Minimum: {np.min(magnitude):.6f}
    Maximum: {np.max(magnitude):.6f}
    Mean: {np.mean(magnitude):.6f}
    Median: {np.median(magnitude):.6f}

Smoothness parameter:
    Alpha = {ALPHA}

Iterations:
    {ITERATIONS}

Note:
    Horn-Schunck estimates dense optical flow using
    brightness constancy and global smoothness regularization.
"""

(RESULTS_DIR / "horn_schunck_summary.txt").write_text(
    summary,
    encoding="utf-8"
)


# --------------------------------------------------
# 14. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 65)

print(f"\nResults saved in:\n{RESULTS_DIR}")

print("\nGenerated files:")
print("  - horn_schunck_flow.png")
print("  - motion_magnitude.png")
print("  - horn_schunck_vectors.png")
print("  - horn_schunck_flow.npy")
print("  - flow_magnitude.npy")
print("  - flow_direction.npy")
print("  - horn_schunck_summary.txt")
