
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_1_PATH = (
    PROJECT_ROOT
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
    / "image_2"
    / "000000_10.png"
)

IMAGE_2_PATH = (
    PROJECT_ROOT
    / "KITTI_Optical_Flow_2025"
    / "data_scene_flow"
    / "training"
    / "image_2"
    / "000000_11.png"
)

RESULT_DIR = PROJECT_ROOT / "Stage_5" / "results" / "Task 2"

INPUT_WIDTH = 400
INPUT_HEIGHT = 120
CORRELATION_LEVELS = 4
CORRELATION_RADIUS = 2
UPDATE_ITERATIONS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# BASIC NETWORK MODULES
# ------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=1
        )
        self.norm1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=1
        )
        self.norm2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        return self.relu(x + residual)


class FeatureEncoder(nn.Module):
    """Shared feature encoder. Output stride = 8."""

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
        )

    def forward(self, x):
        return self.layers(x)


class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
        )

        self.hidden_head = nn.Conv2d(128, 128, 3, padding=1)
        self.context_head = nn.Conv2d(128, 128, 3, padding=1)

    def forward(self, image):
        features = self.encoder(image)

        hidden = torch.tanh(self.hidden_head(features))
        context = F.relu(self.context_head(features))

        return hidden, context


# ------------------------------------------------------------
# ALL-PAIRS CORRELATION PYRAMID
# ------------------------------------------------------------

class CorrelationPyramid:
    def __init__(self, feature1, feature2, levels=4):
        batch, channels, height, width = feature1.shape

        # Correlation between every source and target feature.
        correlation = torch.einsum(
            "bchw,bcij->bhwij",
            feature1,
            feature2
        )

        correlation = correlation / np.sqrt(channels)

        correlation = correlation.reshape(
            batch * height * width,
            1,
            height,
            width
        )

        self.pyramid = [correlation]

        for _ in range(1, levels):
            correlation = F.avg_pool2d(
                correlation,
                kernel_size=2,
                stride=2
            )
            self.pyramid.append(correlation)

    
    def lookup(self, coordinates, radius=2):
        """
        coordinates: [B, 2, H, W]
        Returns: [B, levels * (2r+1)^2, H, W]
        """

        batch, _, height, width = coordinates.shape
        num_locations = batch * height * width

        # Create the 5x5 neighbourhood offsets.
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=coordinates.device),
            torch.arange(-radius, radius + 1, device=coordinates.device),
            indexing="ij"
        )

        offsets = torch.stack(
            [offsets_x, offsets_y],
            dim=-1
        ).float().reshape(1, 25, 2)

        # Flatten source coordinates.
        coords = coordinates.permute(0, 2, 3, 1)
        coords = coords.reshape(num_locations, 1, 2)

        outputs = []

        for level, correlation in enumerate(self.pyramid):

            corr_height, corr_width = correlation.shape[-2:]

            # Scale coordinates for the current pyramid level.
            level_coords = coords / (2 ** level)

            # Generate 25 sampling locations for every source pixel.
            sample_coords = level_coords + offsets

            # Normalize coordinates to [-1, 1].
            grid_x = (
                2.0 * sample_coords[..., 0] /
                max(corr_width - 1, 1)
            ) - 1.0

            grid_y = (
                2.0 * sample_coords[..., 1] /
                max(corr_height - 1, 1)
            ) - 1.0

            # Exact shape: [B*H*W, 1, 25, 2]
            grid = torch.stack(
                [grid_x, grid_y],
                dim=-1
            ).reshape(num_locations, 1, 25, 2)

            # Input and grid now have matching batch dimensions.
            sampled = F.grid_sample(
                correlation,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True
            )

            # [B*H*W, 1, 1, 25] -> [B, 25, H, W]
            sampled = sampled.reshape(
                batch, height, width, 25
            )

            sampled = sampled.permute(0, 3, 1, 2)

            outputs.append(sampled)

        return torch.cat(outputs, dim=1)
    
# ------------------------------------------------------------
# MOTION ENCODER AND RECURRENT UPDATE
# ------------------------------------------------------------

class MotionEncoder(nn.Module):
    def __init__(self, correlation_channels):
        super().__init__()

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(correlation_channels, 96, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, correlation, flow):
        corr_features = self.corr_encoder(correlation)
        flow_features = self.flow_encoder(flow)

        combined = torch.cat(
            [corr_features, flow_features],
            dim=1
        )

        return self.output(combined)


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=256):
        super().__init__()

        self.convz = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1
        )
        self.convr = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1
        )
        self.convq = nn.Conv2d(
            hidden_dim + input_dim, hidden_dim, 3, padding=1
        )

    def forward(self, hidden, inputs):
        combined = torch.cat([hidden, inputs], dim=1)

        update_gate = torch.sigmoid(self.convz(combined))
        reset_gate = torch.sigmoid(self.convr(combined))

        candidate_input = torch.cat(
            [reset_gate * hidden, inputs],
            dim=1
        )

        candidate = torch.tanh(self.convq(candidate_input))

        return (
            (1 - update_gate) * hidden
            + update_gate * candidate
        )


class FlowHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, 3, padding=1)
        )

    def forward(self, hidden):
        return self.layers(hidden)


class UpsamplingMaskHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 8 * 8 * 9, 1)
        )

    def forward(self, hidden):
        return 0.25 * self.layers(hidden)


# ------------------------------------------------------------
# RAFT-STYLE MODEL
# ------------------------------------------------------------

class RAFTForwardModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_encoder = FeatureEncoder()
        self.context_encoder = ContextEncoder()

        correlation_channels = (
            CORRELATION_LEVELS
            * (2 * CORRELATION_RADIUS + 1) ** 2
        )

        self.motion_encoder = MotionEncoder(
            correlation_channels
        )

        self.update_block = ConvGRU(
            hidden_dim=128,
            input_dim=256
        )

        self.flow_head = FlowHead()
        self.mask_head = UpsamplingMaskHead()

    @staticmethod
    def coordinate_grid(batch, height, width, device):
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij"
        )

        grid = torch.stack([x, y], dim=0).float()
        return grid.unsqueeze(0).repeat(batch, 1, 1, 1)

    @staticmethod
    def convex_upsample(flow, mask):
        batch, _, height, width = flow.shape

        mask = mask.view(
            batch, 1, 9, 8, 8, height, width
        )
        mask = torch.softmax(mask, dim=2)

        flow_patches = F.unfold(
            8.0 * flow,
            kernel_size=3,
            padding=1
        )

        flow_patches = flow_patches.view(
            batch, 2, 9, 1, 1, height, width
        )

        upsampled = torch.sum(
            mask * flow_patches,
            dim=2
        )

        upsampled = upsampled.permute(
            0, 1, 4, 2, 5, 3
        )

        return upsampled.reshape(
            batch, 2, height * 8, width * 8
        )

    def forward(self, image1, image2, iterations=5):
        feature1 = self.feature_encoder(image1)
        feature2 = self.feature_encoder(image2)

        hidden, context = self.context_encoder(image1)

        correlation = CorrelationPyramid(
            feature1,
            feature2,
            levels=CORRELATION_LEVELS
        )

        batch, _, height, width = feature1.shape

        coords0 = self.coordinate_grid(
            batch, height, width, image1.device
        )

        coords1 = coords0.clone()
        flow_predictions = []

        for iteration in range(iterations):
            flow = coords1 - coords0

            correlation_features = correlation.lookup(
                coords1,
                radius=CORRELATION_RADIUS
            )

            motion_features = self.motion_encoder(
                correlation_features,
                flow
            )

            update_input = torch.cat(
                [context, motion_features],
                dim=1
            )

            hidden = self.update_block(
                hidden,
                update_input
            )

            delta_flow = self.flow_head(hidden)
            coords1 = coords1 + delta_flow

            flow = coords1 - coords0

            mask = self.mask_head(hidden)
            full_resolution_flow = self.convex_upsample(
                flow,
                mask
            )

            flow_predictions.append(
                full_resolution_flow
            )

            print(
                f"Iteration {iteration + 1}/{iterations}: "
                f"low-resolution flow {tuple(flow.shape)}, "
                f"upsampled flow {tuple(full_resolution_flow.shape)}"
            )

        return flow_predictions


# ------------------------------------------------------------
# IMAGE AND VISUALIZATION UTILITIES
# ------------------------------------------------------------

def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    original_height, original_width = image.shape[:2]

    resized = cv2.resize(
        image,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    )

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return tensor, (original_width, original_height)


def flow_to_color(flow):
    u = flow[..., 0]
    v = flow[..., 1]

    magnitude, angle = cv2.cartToPolar(
        u.astype(np.float32),
        v.astype(np.float32),
        angleInDegrees=True
    )

    hsv = np.zeros(
        (flow.shape[0], flow.shape[1], 3),
        dtype=np.uint8
    )

    hsv[..., 0] = (angle / 2).astype(np.uint8)
    hsv[..., 1] = 255

    if magnitude.max() > 0:
        hsv[..., 2] = np.clip(
            magnitude / magnitude.max() * 255,
            0,
            255
        ).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), magnitude


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("STAGE 5 - TASK 2: RAFT FORWARD PASS")
    print("=" * 65)

    print(f"Device: {DEVICE}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    image1, original_size = load_image(IMAGE_1_PATH)
    image2, _ = load_image(IMAGE_2_PATH)

    image1 = image1.to(DEVICE)
    image2 = image2.to(DEVICE)

    print(f"Input tensor 1: {tuple(image1.shape)}")
    print(f"Input tensor 2: {tuple(image2.shape)}")
    print(f"Original image size: {original_size}")

    model = RAFTForwardModel().to(DEVICE)
    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    start_time = time.time()

    with torch.no_grad():
        predictions = model(
            image1,
            image2,
            iterations=UPDATE_ITERATIONS
        )

    elapsed_time = time.time() - start_time

    final_flow = predictions[-1][0].permute(
        1, 2, 0
    ).cpu().numpy()

    # Resize the flow field to the original KITTI dimensions.
    original_width, original_height = original_size

    final_flow = cv2.resize(
        final_flow,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )

    final_flow[..., 0] *= original_width / INPUT_WIDTH
    final_flow[..., 1] *= original_height / INPUT_HEIGHT

    flow_color, magnitude = flow_to_color(final_flow)

    np.save(
        RESULT_DIR / "final_flow.npy",
        final_flow.astype(np.float32)
    )

    cv2.imwrite(
        str(RESULT_DIR / "final_flow_color.png"),
        flow_color
    )

    magnitude_visualization = np.clip(
        magnitude / max(float(magnitude.max()), 1e-6) * 255,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        str(RESULT_DIR / "flow_magnitude.png"),
        magnitude_visualization
    )

    # Save each iteration's full-resolution flow.
    iteration_statistics = []

    for index, prediction in enumerate(predictions):
        flow = prediction[0].permute(
            1, 2, 0
        ).cpu().numpy()

        flow = cv2.resize(
            flow,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR
        )

        flow[..., 0] *= original_width / INPUT_WIDTH
        flow[..., 1] *= original_height / INPUT_HEIGHT

        np.save(
            RESULT_DIR / f"iteration_{index + 1}_flow.npy",
            flow.astype(np.float32)
        )

        iteration_magnitude = np.sqrt(
            flow[..., 0] ** 2 + flow[..., 1] ** 2
        )

        iteration_statistics.append(
            (
                index + 1,
                float(iteration_magnitude.mean()),
                float(iteration_magnitude.max())
            )
        )

    summary_path = RESULT_DIR / "forward_pass_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("STAGE 5 - TASK 2: RAFT FORWARD PASS\n")
        file.write("=" * 50 + "\n")
        file.write(f"Device: {DEVICE}\n")
        file.write(f"Input size: {INPUT_WIDTH} x {INPUT_HEIGHT}\n")
        file.write(
            f"Original size: {original_width} x {original_height}\n"
        )
        file.write(f"Model parameters: {parameter_count:,}\n")
        file.write(f"Correlation levels: {CORRELATION_LEVELS}\n")
        file.write(f"Correlation radius: {CORRELATION_RADIUS}\n")
        file.write(f"Update iterations: {UPDATE_ITERATIONS}\n")
        file.write(f"Forward pass time: {elapsed_time:.4f} seconds\n")
        file.write(
            f"Final flow shape: {final_flow.shape}\n"
        )
        file.write(
            f"Mean flow magnitude: {magnitude.mean():.6f}\n"
        )
        file.write(
            f"Maximum flow magnitude: {magnitude.max():.6f}\n"
        )
        file.write("\nIteration statistics\n")

        for iteration, mean_mag, max_mag in iteration_statistics:
            file.write(
                f"Iteration {iteration}: "
                f"mean={mean_mag:.6f}, max={max_mag:.6f}\n"
            )

        file.write(
            "\nNote: The model uses randomly initialized weights. "
            "These outputs validate the forward-pass pipeline only; "
            "they are not trained optical-flow predictions.\n"
        )

    print("\nForward pass completed.")
    print(f"Final flow shape: {final_flow.shape}")
    print(f"Mean flow magnitude: {magnitude.mean():.6f}")
    print(f"Maximum flow magnitude: {magnitude.max():.6f}")
    print(f"Forward pass time: {elapsed_time:.4f} seconds")
    print(f"Results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()
