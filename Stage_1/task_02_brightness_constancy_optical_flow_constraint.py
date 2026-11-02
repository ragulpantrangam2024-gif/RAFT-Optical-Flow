
"""
Stage 1 - Task 2
Brightness Constancy, Optical Flow Constraint Equation,
and the Aperture Problem.

This script:
1. Loads two consecutive MPI Sintel frames.
2. Converts them to grayscale.
3. Calculates spatial gradients Ix and Iy.
4. Calculates temporal gradient It.
5. Visualizes the three derivatives.
6. Illustrates the aperture problem.
7. Saves all results inside results/Task 2.

Note:
This task derives and investigates the optical flow constraint
equation. It does not calculate a complete optical flow field.
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

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

RESULTS_DIR = SCRIPT_DIR / "results" / "Task 2"


# ============================================================
# 2. IMAGE LOADING
# ============================================================

def load_grayscale_image(image_path):
    """Load an image directly in grayscale."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    return image


# ============================================================
# 3. BRIGHTNESS CONSTANCY AND IMAGE DERIVATIVES
# ============================================================

def calculate_image_derivatives(frame1, frame2):
    """
    Calculate spatial and temporal intensity derivatives.

    Ix = horizontal intensity gradient
    Iy = vertical intensity gradient
    It = temporal intensity gradient
    """

    # Convert images to float32 for numerical calculations
    frame1_float = frame1.astype(np.float32) / 255.0
    frame2_float = frame2.astype(np.float32) / 255.0

    # Spatial gradient in horizontal direction
    Ix = cv2.Sobel(
        frame1_float,
        cv2.CV_32F,
        1, 0,
        ksize=3
    ) / 8.0

    # Spatial gradient in vertical direction
    Iy = cv2.Sobel(
        frame1_float,
        cv2.CV_32F,
        0, 1,
        ksize=3
    ) / 8.0

    # Temporal gradient between consecutive frames
    It = frame2_float - frame1_float

    return Ix, Iy, It


# ============================================================
# 4. VISUALIZE IMAGE DERIVATIVES
# ============================================================

def visualize_derivatives(
    frame1,
    frame2,
    Ix,
    Iy,
    It
):
    """Display and save the derivative visualizations."""

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 9)
    )

    axes[0, 0].imshow(frame1, cmap="gray")
    axes[0, 0].set_title("Frame 1")

    axes[0, 1].imshow(frame2, cmap="gray")
    axes[0, 1].set_title("Frame 2")

    axes[0, 2].imshow(
        Ix,
        cmap="gray",
        vmin=-0.2,
        vmax=0.2
    )
    axes[0, 2].set_title("Horizontal Gradient (Ix)")

    axes[1, 0].imshow(
        Iy,
        cmap="gray",
        vmin=-0.2,
        vmax=0.2
    )
    axes[1, 0].set_title("Vertical Gradient (Iy)")

    axes[1, 1].imshow(
        It,
        cmap="gray",
        vmin=-0.2,
        vmax=0.2
    )
    axes[1, 1].set_title("Temporal Gradient (It)")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.05,
        0.85,
        "Optical Flow Constraint Equation",
        fontsize=13,
        weight="bold"
    )

    axes[1, 2].text(
        0.05,
        0.65,
        r"$I_xu + I_yv + I_t = 0$",
        fontsize=20
    )

    axes[1, 2].text(
        0.05,
        0.40,
        "One equation\nTwo unknowns (u, v)",
        fontsize=14
    )

    axes[1, 2].text(
        0.05,
        0.15,
        "Additional constraints are required.",
        fontsize=11
    )

    for axis in axes.flat:
        if axis.axison:
            axis.axis("off")

    fig.suptitle(
        "Task 2: Brightness Constancy and Image Derivatives",
        fontsize=16
    )

    fig.tight_layout()

    output_path = RESULTS_DIR / "image_derivatives.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# 5. APERTURE PROBLEM ILLUSTRATION
# ============================================================

def visualize_aperture_problem():
    """
    Create a conceptual illustration of the aperture problem.

    A straight edge has an observable normal direction,
    but motion parallel to the edge is ambiguous locally.
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)

    # Draw a small observation window
    window = Rectangle(
        (3, 2),
        4,
        3,
        linewidth=2,
        edgecolor="black",
        facecolor="none"
    )

    ax.add_patch(window)

    # Draw a diagonal edge
    x = np.linspace(1, 9, 100)
    y = 0.65 * x + 0.7

    ax.plot(
        x,
        y,
        linewidth=3,
        label="Moving straight edge"
    )

    # Motion component perpendicular to edge
    ax.annotate(
        "",
        xy=(5.8, 4.5),
        xytext=(5.0, 3.3),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2
        )
    )

    ax.text(
        6.0,
        4.5,
        "Observable normal motion",
        fontsize=10
    )

    # Motion component parallel to edge
    ax.annotate(
        "",
        xy=(6.7, 3.7),
        xytext=(5.5, 2.9),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=2
        )
    )

    ax.text(
        6.8,
        3.2,
        "Tangential motion\nis ambiguous",
        fontsize=10
    )

    ax.set_title(
        "Aperture Problem: Local Motion Ambiguity",
        fontsize=14
    )

    ax.text(
        3,
        1.4,
        "Observation window",
        fontsize=11
    )

    ax.axis("off")

    fig.tight_layout()

    output_path = RESULTS_DIR / "aperture_problem.png"

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return output_path


# ============================================================
# 6. NUMERICAL ANALYSIS
# ============================================================

def print_derivative_statistics(Ix, Iy, It):
    """Print basic statistics of the image derivatives."""

    print("\nIMAGE DERIVATIVE STATISTICS")
    print("-" * 45)

    derivatives = {
        "Ix (horizontal)": Ix,
        "Iy (vertical)": Iy,
        "It (temporal)": It
    }

    for name, derivative in derivatives.items():
        print(f"\n{name}")
        print(f"  Minimum : {np.min(derivative):.6f}")
        print(f"  Maximum : {np.max(derivative):.6f}")
        print(f"  Mean    : {np.mean(derivative):.6f}")
        print(f"  Std Dev : {np.std(derivative):.6f}")


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("STAGE 1 - TASK 2")
    print("BRIGHTNESS CONSTANCY AND OPTICAL FLOW CONSTRAINT")
    print("=" * 60)

    # Create Task 2 results directory
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load frames
    frame1 = load_grayscale_image(FRAME1_PATH)
    frame2 = load_grayscale_image(FRAME2_PATH)

    if frame1.shape != frame2.shape:
        raise ValueError(
            "Input frames must have identical dimensions."
        )

    height, width = frame1.shape

    print(f"\nImage dimensions: {width} x {height}")

    # Calculate derivatives
    Ix, Iy, It = calculate_image_derivatives(
        frame1,
        frame2
    )

    # Save numerical derivative arrays
    np.save(RESULTS_DIR / "Ix_horizontal_gradient.npy", Ix)
    np.save(RESULTS_DIR / "Iy_vertical_gradient.npy", Iy)
    np.save(RESULTS_DIR / "It_temporal_gradient.npy", It)

    # Generate visualizations
    derivative_path = visualize_derivatives(
        frame1,
        frame2,
        Ix,
        Iy,
        It
    )

    aperture_path = visualize_aperture_problem()

    # Print statistics
    print_derivative_statistics(Ix, Iy, It)

    print("\n" + "=" * 60)
    print("TASK 2 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nResults saved in:\n{RESULTS_DIR}")

    print("\nGenerated files:")
    print(f"  - {derivative_path.name}")
    print(f"  - {aperture_path.name}")
    print("  - Ix_horizontal_gradient.npy")
    print("  - Iy_vertical_gradient.npy")
    print("  - It_temporal_gradient.npy")

    print(
        "\nNote: The derivatives are inputs to the optical flow "
        "constraint equation. This task does not solve for u and v."
    )


if __name__ == "__main__":
    main()
