
"""
Task 1: Optical Flow Fundamentals — Inspect Consecutive Frames

Loads two consecutive MPI Sintel frames and saves a comparison
visualization and absolute pixel-difference image.

Note: Pixel differences are NOT optical-flow estimates.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# ============================================================
# 1. DEFINE PATHS
# ============================================================

# Directory containing this Python script: RAFT_Optical_Flow/Stage_1
SCRIPT_DIR = Path(__file__).resolve().parent

# Root directory of the RAFT project
PROJECT_DIR = SCRIPT_DIR.parent

# Input images from the MPI Sintel dataset
FRAME1_PATH = (
    PROJECT_DIR
    / "MPI-Sintel-training_images"
    / "training"
    / "clean"
    / "alley_1"
    / "frame_0001.png"
)

FRAME2_PATH = (
    PROJECT_DIR
    / "MPI-Sintel-training_images"
    / "training"
    / "clean"
    / "alley_1"
    / "frame_0002.png"
)

# Task 1 has its own results folder
RESULTS_DIR = SCRIPT_DIR / "results" / "Task 1"


# ============================================================
# 2. LOAD IMAGES
# ============================================================

def load_image(image_path):
    """Load an image and report a clear error if it is missing."""

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}\n\n"
            "Check the dataset path and image filename."
        )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(
            f"OpenCV could not read the image:\n{image_path}"
        )

    # Convert BGR (OpenCV format) to RGB (Matplotlib format)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ============================================================
# 3. MAIN TASK
# ============================================================

def main():

    print("=" * 60)
    print("TASK 1: OPTICAL FLOW FUNDAMENTALS")
    print("=" * 60)

    print(f"\nLoading Frame 1:\n{FRAME1_PATH}")
    print(f"\nLoading Frame 2:\n{FRAME2_PATH}")

    frame1 = load_image(FRAME1_PATH)
    frame2 = load_image(FRAME2_PATH)

    # Both frames must have matching dimensions
    if frame1.shape != frame2.shape:
        raise ValueError(
            "The two images have different dimensions.\n"
            f"Frame 1: {frame1.shape}\n"
            f"Frame 2: {frame2.shape}"
        )

    height, width = frame1.shape[:2]

    print(f"\nImage dimensions: {width} x {height}")

    # ========================================================
    # 4. CALCULATE ABSOLUTE PIXEL DIFFERENCE
    # ========================================================

    # Shows pixel-wise changes between the two images.
    # This is NOT an optical-flow calculation.
    difference = cv2.absdiff(frame1, frame2)

    # ========================================================
    # 5. CREATE TASK-SPECIFIC RESULTS FOLDER
    # ========================================================

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save each input frame and the difference image
    frame1_path = RESULTS_DIR / "frame_0001.png"
    frame2_path = RESULTS_DIR / "frame_0002.png"
    difference_path = RESULTS_DIR / "absolute_pixel_difference.png"

    plt.imsave(str(frame1_path), frame1)
    plt.imsave(str(frame2_path), frame2)
    plt.imsave(str(difference_path), difference)

    # ========================================================
    # 6. CREATE COMBINED COMPARISON FIGURE
    # ========================================================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(frame1)
    axes[0].set_title("Frame 1")

    axes[1].imshow(frame2)
    axes[1].set_title("Frame 2")

    axes[2].imshow(difference)
    axes[2].set_title("Absolute Pixel Difference")

    for axis in axes:
        axis.axis("off")

    fig.suptitle("MPI Sintel — alley_1 Frame Comparison")
    fig.tight_layout()

    comparison_path = RESULTS_DIR / "image_pair_comparison.png"

    fig.savefig(
        str(comparison_path),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ========================================================
    # 7. REPORT RESULTS
    # ========================================================

    print("\nTask 1 completed successfully!")
    print(f"Image dimensions: {width} x {height}")

    print(f"\nAll results saved in:\n{RESULTS_DIR}")

    print("\nGenerated files:")

    for output_file in (
        frame1_path,
        frame2_path,
        difference_path,
        comparison_path,
    ):
        print(f"  - {output_file.name}")

    print(
        "\nNote: Pixel differences show image changes, "
        "not optical-flow estimates."
    )


if __name__ == "__main__":
    main()
