
"""
STAGE 4 - TASK 6: ITERATIVE OPTICAL FLOW REFINEMENT

A simplified RAFT-inspired architecture:
1. Extract CNN features from two KITTI images.
2. Build an all-pairs correlation pyramid.
3. Look up local correlations around the current flow estimate.
4. Use a recurrent update block to predict flow corrections.
5. Repeat the update for multiple iterations.

This is an educational architecture demonstration.
The network is randomly initialized and is NOT trained RAFT.
"""

from pathlib import Path
import csv
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
    / "image_2"
)

OUTPUT_DIR = PROJECT_DIR / "Stage_4" / "results" / "Task 6"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE1_PATH = DATASET_DIR / "000000_10.png"
IMAGE2_PATH = DATASET_DIR / "000000_11.png"

FEATURE_HEIGHT = 24
FEATURE_WIDTH = 40
FEATURE_CHANNELS = 64

PYRAMID_LEVELS = 4
LOOKUP_RADIUS = 2
UPDATE_ITERATIONS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 2. CNN FEATURE ENCODER
# ============================================================

class FeatureEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, FEATURE_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, image):

        features = self.network(image)

        features = F.adaptive_avg_pool2d(
            features,
            (FEATURE_HEIGHT, FEATURE_WIDTH)
        )

        return F.normalize(features, p=2, dim=1)


# ============================================================
# 3. ALL-PAIRS CORRELATION PYRAMID
# ============================================================

def build_correlation_pyramid(feature1, feature2):

    batch, channels, height, width = feature1.shape

    locations = height * width

    feature1 = feature1.reshape(
        batch, channels, locations
    )

    feature2 = feature2.reshape(
        batch, channels, locations
    )

    correlation = torch.bmm(
        feature1.transpose(1, 2),
        feature2
    )

    correlation = correlation.reshape(
        batch, height, width, height, width
    )

    # Treat every source location as a separate
    # correlation map over target-image locations.
    correlation_maps = correlation.reshape(
        batch * locations, 1, height, width
    )

    pyramid = [correlation_maps]

    for _ in range(1, PYRAMID_LEVELS):

        previous = pyramid[-1]

        if previous.shape[-2] < 2 or previous.shape[-1] < 2:
            break

        pyramid.append(
            F.avg_pool2d(previous, kernel_size=2, stride=2)
        )

    return pyramid


# ============================================================
# 4. MULTI-SCALE LOCAL CORRELATION LOOKUP
# ============================================================

def correlation_lookup(pyramid, coordinates):

    batch, height, width, _ = coordinates.shape

    offsets = torch.arange(
        -LOOKUP_RADIUS,
        LOOKUP_RADIUS + 1,
        device=coordinates.device,
        dtype=coordinates.dtype
    )

    dy, dx = torch.meshgrid(
        offsets,
        offsets,
        indexing="ij"
    )

    offset_grid = torch.stack(
        [dx.reshape(-1), dy.reshape(-1)],
        dim=-1
    )

    lookup_results = []

    for level, correlation_maps in enumerate(pyramid):

        target_height = correlation_maps.shape[-2]
        target_width = correlation_maps.shape[-1]

        scale = 2 ** level

        scaled_coordinates = coordinates / scale

        candidate_coordinates = (
            scaled_coordinates.unsqueeze(-2)
            + offset_grid.view(1, 1, 1, -1, 2)
        )

        x = candidate_coordinates[..., 0]
        y = candidate_coordinates[..., 1]

        # Convert coordinates to grid_sample range [-1, 1].
        normalized_x = (
            2.0 * x / max(target_width - 1, 1) - 1.0
        )

        normalized_y = (
            2.0 * y / max(target_height - 1, 1) - 1.0
        )

        sampling_grid = torch.stack(
            [normalized_x, normalized_y],
            dim=-1
        )

        sampling_grid = sampling_grid.reshape(
            batch * height * width,
            1,
            -1,
            2
        )

        sampled = F.grid_sample(
            correlation_maps,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        sampled = sampled.reshape(
            batch, height, width, -1
        )

        lookup_results.append(sampled)

    return torch.cat(lookup_results, dim=-1)


# ============================================================
# 5. CONVGRU-STYLE RECURRENT UPDATE BLOCK
# ============================================================

class RecurrentUpdateBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden_channels = 64

        # Correlation: 4 levels * 25 values = 100
        # Flow: 2 channels
        # Context: 64 channels
        self.input_channels = 100 + 2 + 64

        combined_channels = (
            self.hidden_channels + self.input_channels
        )

        self.gates = nn.Conv2d(
            combined_channels,
            2 * self.hidden_channels,
            kernel_size=3,
            padding=1
        )

        self.candidate = nn.Conv2d(
            combined_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )

    def forward(self, hidden, update_input):

        combined = torch.cat(
            [hidden, update_input],
            dim=1
        )

        update_gate, reset_gate = torch.sigmoid(
            self.gates(combined)
        ).chunk(2, dim=1)

        candidate_input = torch.cat(
            [reset_gate * hidden, update_input],
            dim=1
        )

        candidate_state = torch.tanh(
            self.candidate(candidate_input)
        )

        hidden = (
            (1.0 - update_gate) * hidden
            + update_gate * candidate_state
        )

        flow_delta = self.flow_head(hidden)

        return hidden, flow_delta


# ============================================================
# 6. IMAGE AND FLOW VISUALIZATION
# ============================================================

def flow_to_rgb(flow):

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    hsv = np.zeros(
        (flow.shape[0], flow.shape[1], 3),
        dtype=np.uint8
    )

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255

    hsv[..., 2] = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )


def save_flow_visualization(flow, filename, title):

    rgb = flow_to_rgb(flow)

    plt.figure(figsize=(12, 5))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150
    )

    plt.close()


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 4 - TASK 6: ITERATIVE FLOW REFINEMENT")
    print("=" * 70)

    print("Processing device:", DEVICE)
    print("Output directory:", OUTPUT_DIR)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print("\nLoading KITTI image pair...")

    image1 = cv2.imread(str(IMAGE1_PATH))
    image2 = cv2.imread(str(IMAGE2_PATH))

    if image1 is None or image2 is None:
        raise FileNotFoundError(
            "Could not load KITTI image pair. "
            "Check the image_2 directory and filenames."
        )

    if image1.shape != image2.shape:
        raise ValueError("Input image dimensions do not match.")

    height, width = image1.shape[:2]

    print("Image dimensions:", width, "x", height)

    def prepare_image(image):

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        tensor = torch.from_numpy(
            image.copy()
        ).permute(2, 0, 1).float() / 255.0

        return tensor.unsqueeze(0).to(DEVICE)

    tensor1 = prepare_image(image1)
    tensor2 = prepare_image(image2)

    # --------------------------------------------------------
    # Initialize networks
    # --------------------------------------------------------

    encoder = FeatureEncoder().to(DEVICE).eval()

    update_block = RecurrentUpdateBlock().to(DEVICE).eval()

    total_parameters = (
        sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in update_block.parameters())
    )

    print("\nTotal network parameters:", total_parameters)

    # --------------------------------------------------------
    # Extract CNN features
    # --------------------------------------------------------

    print("\nExtracting CNN features...")

    start = time.perf_counter()

    with torch.no_grad():

        feature1 = encoder(tensor1)
        feature2 = encoder(tensor2)

    extraction_time = time.perf_counter() - start

    batch, channels, feature_height, feature_width = feature1.shape

    print("Feature 1:", tuple(feature1.shape))
    print("Feature 2:", tuple(feature2.shape))
    print(f"Feature extraction time: {extraction_time:.4f} seconds")

    # --------------------------------------------------------
    # Build correlation pyramid
    # --------------------------------------------------------

    print("\nBuilding correlation pyramid...")

    with torch.no_grad():

        pyramid = build_correlation_pyramid(
            feature1,
            feature2
        )

    for level, corr in enumerate(pyramid):

        print(
            f"Level {level}: "
            f"{tuple(corr.shape)}"
        )

    if len(pyramid) != PYRAMID_LEVELS:
        raise RuntimeError("Unexpected number of pyramid levels.")

    # --------------------------------------------------------
    # Initialize coordinates and flow
    # --------------------------------------------------------

    print("\nInitializing flow and hidden state...")

    y_grid, x_grid = torch.meshgrid(
        torch.arange(feature_height, device=DEVICE),
        torch.arange(feature_width, device=DEVICE),
        indexing="ij"
    )

    coordinates = torch.stack(
        [x_grid, y_grid],
        dim=-1
    ).float().unsqueeze(0)

    initial_flow = torch.zeros(
        1, 2, feature_height, feature_width,
        device=DEVICE
    )

    hidden = torch.zeros(
        1, 64, feature_height, feature_width,
        device=DEVICE
    )

    # Context features are derived from the first image.
    context = feature1

    current_flow = initial_flow.clone()

    iteration_results = []
    metrics = []

    # --------------------------------------------------------
    # Iterative flow refinement
    # --------------------------------------------------------

    print("\nStarting iterative refinement...")

    start = time.perf_counter()

    with torch.no_grad():

        for iteration in range(UPDATE_ITERATIONS):

            # Current target coordinates.
            current_coordinates = (
                coordinates
                + current_flow.permute(0, 2, 3, 1)
            )

            # Local correlation lookup.
            corr_features = correlation_lookup(
                pyramid,
                current_coordinates
            )

            # Prepare recurrent input.
            update_input = torch.cat(
                [
                    corr_features.permute(0, 3, 1, 2),
                    current_flow,
                    context
                ],
                dim=1
            )

            # Recurrent update.
            hidden, flow_delta = update_block(
                hidden,
                update_input
            )

            # Refine the flow estimate.
            current_flow = current_flow + flow_delta

            magnitude = torch.sqrt(
                current_flow[:, 0] ** 2
                + current_flow[:, 1] ** 2
            )

            mean_magnitude = magnitude.mean().item()
            max_magnitude = magnitude.max().item()

            flow_numpy = current_flow[0].permute(
                1, 2, 0
            ).cpu().numpy()

            iteration_results.append(
                flow_numpy.copy()
            )

            metrics.append(
                {
                    "iteration": iteration + 1,
                    "mean_flow_magnitude": mean_magnitude,
                    "maximum_flow_magnitude": max_magnitude
                }
            )

            print(
                f"Iteration {iteration + 1}: "
                f"Mean magnitude = {mean_magnitude:.6f}, "
                f"Maximum = {max_magnitude:.6f}"
            )

    refinement_time = time.perf_counter() - start

    # --------------------------------------------------------
    # Save iteration visualizations
    # --------------------------------------------------------

    print("\nSaving iteration visualizations...")

    for index, flow in enumerate(iteration_results):

        full_resolution_flow = cv2.resize(
            flow,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # Scale displacement vectors to original image pixels.
        full_resolution_flow[..., 0] *= width / feature_width
        full_resolution_flow[..., 1] *= height / feature_height

        save_flow_visualization(
            full_resolution_flow,
            f"flow_iteration_{index + 1}.png",
            f"Flow Estimate - Iteration {index + 1}"
        )

        np.save(
            OUTPUT_DIR / f"flow_iteration_{index + 1}.npy",
            full_resolution_flow
        )

    # --------------------------------------------------------
    # Compare iterations
    # --------------------------------------------------------

    print("\nCreating iteration comparison...")

    figure, axes = plt.subplots(
        1,
        UPDATE_ITERATIONS,
        figsize=(20, 5)
    )

    for index, (axis, flow) in enumerate(
        zip(axes, iteration_results)
    ):

        full_flow = cv2.resize(
            flow,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        full_flow[..., 0] *= width / feature_width
        full_flow[..., 1] *= height / feature_height

        axis.imshow(flow_to_rgb(full_flow))
        axis.set_title(f"Iteration {index + 1}")
        axis.axis("off")

    plt.suptitle("Iterative Optical Flow Refinement")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "iteration_comparison.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Save numerical metrics
    # --------------------------------------------------------

    csv_path = OUTPUT_DIR / "iteration_metrics.csv"

    with open(csv_path, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "iteration",
                "mean_flow_magnitude",
                "maximum_flow_magnitude"
            ]
        )

        writer.writeheader()
        writer.writerows(metrics)

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR / "iterative_refinement_summary.txt"
    )

    with open(summary_path, "w") as file:

        file.write("STAGE 4 - TASK 6: ITERATIVE FLOW REFINEMENT\n")
        file.write("=" * 65 + "\n\n")

        file.write(f"Image dimensions: {width} x {height}\n")
        file.write(
            f"Feature dimensions: "
            f"{feature_height} x {feature_width}\n"
        )

        file.write(f"Pyramid levels: {len(pyramid)}\n")
        file.write(f"Lookup radius: {LOOKUP_RADIUS}\n")
        file.write(f"Iterations: {UPDATE_ITERATIONS}\n")

        file.write(
            f"Feature extraction time: "
            f"{extraction_time:.4f} seconds\n"
        )

        file.write(
            f"Refinement time: "
            f"{refinement_time:.4f} seconds\n\n"
        )

        file.write("ITERATION STATISTICS\n")
        file.write("-" * 65 + "\n")

        for item in metrics:

            file.write(
                f"Iteration {item['iteration']}: "
                f"Mean magnitude = "
                f"{item['mean_flow_magnitude']:.6f}, "
                f"Maximum = "
                f"{item['maximum_flow_magnitude']:.6f}\n"
            )

        file.write("\nIMPORTANT LIMITATION\n")
        file.write(
            "The encoder and recurrent update block use random "
            "initialization. The flow estimates are not trained "
            "optical-flow predictions. This experiment demonstrates "
            "the iterative refinement architecture only.\n"
        )

    print("\n" + "=" * 70)
    print("TASK 6 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("Refinement time:", f"{refinement_time:.4f} seconds")
    print("Results saved to:", OUTPUT_DIR)

    print("\nGenerated files:")
    print("- flow_iteration_1.png through flow_iteration_5.png")
    print("- flow_iteration_1.npy through flow_iteration_5.npy")
    print("- iteration_comparison.png")
    print("- iteration_metrics.csv")
    print("- iterative_refinement_summary.txt")


if __name__ == "__main__":
    main()
