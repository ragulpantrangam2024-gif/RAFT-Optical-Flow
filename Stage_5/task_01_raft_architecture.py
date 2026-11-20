
"""
STAGE 5 - TASK 1: RAFT ARCHITECTURE INTEGRATION

Implements and tests the main RAFT building blocks:
1. Feature encoder
2. Context encoder
3. All-pairs correlation pyramid
4. Motion encoder
5. ConvGRU update block
6. Flow prediction head
7. Convex upsampling mask head

This task validates architecture components only.
It does not train the network or produce trained optical flow.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results",
    "Task 1"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 128
CONTEXT_DIM = 128
FEATURE_DIM = 128
CORRELATION_LEVELS = 4
CORRELATION_RADIUS = 2

print("=" * 70)
print("STAGE 5 - TASK 1: RAFT ARCHITECTURE INTEGRATION")
print("=" * 70)
print("Processing device:", DEVICE)


# ------------------------------------------------------------
# 2. BASIC RESIDUAL BLOCK
# ------------------------------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels, 3,
            stride=stride, padding=1
        )
        self.norm1 = nn.GroupNorm(8, channels)

        self.conv2 = nn.Conv2d(
            channels, channels, 3,
            padding=1
        )
        self.norm2 = nn.GroupNorm(8, channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    channels, channels, 1,
                    stride=stride
                ),
                nn.GroupNorm(8, channels)
            )

    def forward(self, x):

        identity = x

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(out + identity)


# ------------------------------------------------------------
# 3. FEATURE ENCODER
# ------------------------------------------------------------

class FeatureEncoder(nn.Module):

    def __init__(self, output_dim=FEATURE_DIM):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, output_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, output_dim),
            nn.ReLU(inplace=True),

            ResidualBlock(output_dim),
            ResidualBlock(output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# 4. CONTEXT ENCODER
# ------------------------------------------------------------

class ContextEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, HIDDEN_DIM + CONTEXT_DIM, 3,
                      stride=2, padding=1),
            nn.GroupNorm(8, HIDDEN_DIM + CONTEXT_DIM),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        output = self.network(x)

        hidden, context = torch.split(
            output,
            [HIDDEN_DIM, CONTEXT_DIM],
            dim=1
        )

        hidden = torch.tanh(hidden)
        context = F.relu(context)

        return hidden, context


# ------------------------------------------------------------
# 5. ALL-PAIRS CORRELATION PYRAMID
# ------------------------------------------------------------

class CorrelationPyramid:

    def __init__(self, feature1, feature2,
                 levels=CORRELATION_LEVELS):

        batch, channels, height, width = feature1.shape

        self.batch = batch
        self.height = height
        self.width = width
        self.levels = []

        feature1 = feature1.reshape(
            batch, channels, height * width
        )

        feature2 = feature2.reshape(
            batch, channels, height * width
        )

        correlation = torch.matmul(
            feature1.transpose(1, 2),
            feature2
        )

        correlation = correlation / (channels ** 0.5)

        correlation = correlation.reshape(
            batch, height, width, 1, height, width
        )

        correlation = correlation.reshape(
            batch * height * width,
            1,
            height,
            width
        )

        self.levels.append(correlation)

        for _ in range(1, levels):

            correlation = F.avg_pool2d(
                correlation,
                kernel_size=2,
                stride=2
            )

            self.levels.append(correlation)


# ------------------------------------------------------------
# 6. MOTION ENCODER
# ------------------------------------------------------------

class MotionEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        correlation_channels = (
            CORRELATION_LEVELS *
            (2 * CORRELATION_RADIUS + 1) ** 2
        )

        self.correlation_encoder = nn.Sequential(
            nn.Conv2d(
                correlation_channels, 96, 1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 64, 7, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Sequential(
            nn.Conv2d(96, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, correlation, flow):

        corr_features = self.correlation_encoder(correlation)
        flow_features = self.flow_encoder(flow)

        combined = torch.cat(
            [corr_features, flow_features],
            dim=1
        )

        return self.output(combined)


# ------------------------------------------------------------
# 7. CONVGRU UPDATE BLOCK
# ------------------------------------------------------------

class ConvGRU(nn.Module):

    def __init__(self, hidden_dim, input_dim):

        super().__init__()

        combined_dim = hidden_dim + input_dim

        self.z = nn.Conv2d(
            combined_dim, hidden_dim, 3, padding=1
        )

        self.r = nn.Conv2d(
            combined_dim, hidden_dim, 3, padding=1
        )

        self.q = nn.Conv2d(
            combined_dim, hidden_dim, 3, padding=1
        )

    def forward(self, hidden, inputs):

        combined = torch.cat([hidden, inputs], dim=1)

        update_gate = torch.sigmoid(self.z(combined))
        reset_gate = torch.sigmoid(self.r(combined))

        candidate_input = torch.cat(
            [reset_gate * hidden, inputs],
            dim=1
        )

        candidate = torch.tanh(self.q(candidate_input))

        hidden = (
            (1 - update_gate) * hidden
            + update_gate * candidate
        )

        return hidden


# ------------------------------------------------------------
# 8. FLOW PREDICTION HEAD
# ------------------------------------------------------------

class FlowHead(nn.Module):

    def __init__(self, input_dim=HIDDEN_DIM):

        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(input_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 2, 3, padding=1)
        )

    def forward(self, hidden):
        return self.network(hidden)


# ------------------------------------------------------------
# 9. CONVEX UPSAMPLING MASK HEAD
# ------------------------------------------------------------

class UpsamplingMaskHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(HIDDEN_DIM, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 8 * 8 * 9, 1)
        )

    def forward(self, hidden):

        mask = self.network(hidden)

        return 0.25 * mask


# ------------------------------------------------------------
# 10. ARCHITECTURE VALIDATION
# ------------------------------------------------------------

def main():

    torch.manual_seed(42)

    # Small test input to validate the architecture.
    # Dimensions are divisible by 8.
    image1 = torch.rand(1, 3, 128, 192).to(DEVICE)
    image2 = torch.rand(1, 3, 128, 192).to(DEVICE)

    feature_encoder = FeatureEncoder().to(DEVICE)
    context_encoder = ContextEncoder().to(DEVICE)

    motion_encoder = MotionEncoder().to(DEVICE)
    update_block = ConvGRU(
        HIDDEN_DIM,
        CONTEXT_DIM + 128
    ).to(DEVICE)

    flow_head = FlowHead().to(DEVICE)
    mask_head = UpsamplingMaskHead().to(DEVICE)

    print("\nExtracting image features...")

    with torch.no_grad():

        feature1 = feature_encoder(image1)
        feature2 = feature_encoder(image2)

        print("Feature 1:", tuple(feature1.shape))
        print("Feature 2:", tuple(feature2.shape))

        print("\nExtracting context features...")

        hidden, context = context_encoder(image1)

        print("Hidden state:", tuple(hidden.shape))
        print("Context:", tuple(context.shape))

        print("\nBuilding correlation pyramid...")

        correlation = CorrelationPyramid(
            feature1,
            feature2
        )

        for level, corr in enumerate(correlation.levels):
            print(
                f"Level {level}:",
                tuple(corr.shape)
            )

        print("\nTesting motion encoder...")

        batch, _, height, width = feature1.shape

        correlation_input = torch.rand(
            batch,
            CORRELATION_LEVELS *
            (2 * CORRELATION_RADIUS + 1) ** 2,
            height,
            width
        ).to(DEVICE)

        flow = torch.zeros(
            batch, 2, height, width
        ).to(DEVICE)

        motion_features = motion_encoder(
            correlation_input,
            flow
        )

        print("Motion features:", tuple(motion_features.shape))

        print("\nTesting recurrent update block...")

        update_input = torch.cat(
            [context, motion_features],
            dim=1
        )

        updated_hidden = update_block(
            hidden,
            update_input
        )

        print("Updated hidden:", tuple(updated_hidden.shape))

        print("\nTesting flow prediction head...")

        flow_prediction = flow_head(updated_hidden)

        print("Flow prediction:", tuple(flow_prediction.shape))

        print("\nTesting upsampling mask head...")

        upsampling_mask = mask_head(updated_hidden)

        print("Upsampling mask:", tuple(upsampling_mask.shape))

        total_parameters = sum(
            parameter.numel()
            for module in [
                feature_encoder,
                context_encoder,
                motion_encoder,
                update_block,
                flow_head,
                mask_head
            ]
            for parameter in module.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for module in [
                feature_encoder,
                context_encoder,
                motion_encoder,
                update_block,
                flow_head,
                mask_head
            ]
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "raft_architecture_summary.txt"
    )

    with open(summary_path, "w", encoding="utf-8") as file:

        file.write("STAGE 5 - TASK 1: RAFT ARCHITECTURE\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Device: {DEVICE}\n")
        file.write(f"Total parameters: {total_parameters:,}\n")
        file.write(f"Trainable parameters: {trainable_parameters:,}\n\n")

        file.write(f"Feature shape: {tuple(feature1.shape)}\n")
        file.write(f"Hidden shape: {tuple(hidden.shape)}\n")
        file.write(f"Context shape: {tuple(context.shape)}\n")

        for level, corr in enumerate(correlation.levels):
            file.write(
                f"Correlation level {level}: {tuple(corr.shape)}\n"
            )

        file.write(
            f"Motion features: {tuple(motion_features.shape)}\n"
        )
        file.write(
            f"Updated hidden: {tuple(updated_hidden.shape)}\n"
        )
        file.write(
            f"Flow prediction: {tuple(flow_prediction.shape)}\n"
        )
        file.write(
            f"Upsampling mask: {tuple(upsampling_mask.shape)}\n"
        )

        file.write("\nArchitecture validation completed.\n")
        file.write(
            "Note: Components are randomly initialized and untrained.\n"
        )

    print("\n" + "=" * 70)
    print("TASK 1 ARCHITECTURE VALIDATION COMPLETED")
    print("=" * 70)

    print("Total parameters:", f"{total_parameters:,}")
    print("Trainable parameters:", f"{trainable_parameters:,}")
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
