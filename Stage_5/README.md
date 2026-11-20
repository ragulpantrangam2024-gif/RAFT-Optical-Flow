Stage 5 – RAFT Architecture, Training and Evaluation

Overview

Stage 5 focuses on implementing an educational RAFT-style optical flow pipeline using PyTorch. The stage covers architecture integration, forward inference, training pipeline development, model training, visualization and quantitative evaluation.

The implementation is inspired by RAFT (Recurrent All-Pairs Field Transforms), which uses feature extraction, all-pairs correlation and recurrent flow refinement.

Note: This is a simplified educational implementation and not an exact reproduction of the original RAFT architecture.

Tasks Completed

Task 1 – RAFT Architecture Integration

Implemented the main components of a simplified RAFT-style architecture:

Feature Encoder

Context Encoder

Correlation Pyramid

Motion Encoder

ConvGRU

Flow Head

Upsampling Mask Head

Validated the connections and tensor dimensions of the individual modules.

Total parameters: 3,325,698.

Task 2 – RAFT Forward Pass

Implemented a complete forward pass using a KITTI image pair.

The model extracts feature maps, constructs correlation information, performs recurrent flow refinement and generates an upsampled optical flow prediction.

Input resolution: 400 × 120

Feature resolution: 50 × 15

Recurrent updates: 5

Model parameters: 3,699,586

Final flow resolution: 1242 × 375

Saved intermediate flow predictions, final flow arrays, colour visualizations and magnitude maps.

Task 3 – RAFT Training Pipeline

Developed the training pipeline using MPI-Sintel and KITTI datasets.

Implemented:

Sintel .flo ground-truth flow loading

KITTI encoded flow PNG decoding

Invalid-pixel masking

Image and flow resizing

Sequence loss

AdamW optimizer

Learning-rate scheduling

Gradient validation

Model checkpoint saving

Dataset samples discovered:

MPI-Sintel Final: 1,041

KITTI: 200

Combined: 1,241

Performed an initial training-pipeline smoke test to verify forward propagation, loss calculation and gradient computation.

Task 4 – RAFT Model Training

Trained the simplified model using Sintel and KITTI samples.

Training configuration:

Device: CPU

Epochs: 3

Batch size: 1

Learning rate: 0.00001

Samples per epoch: 100

Recurrent updates: 5

Optimizer: AdamW

Best validation loss: 9.713842 (Epoch 1).

Validation loss remained nearly flat throughout training, indicating that the short training run did not demonstrate strong learning progress.

Saved the best and latest model checkpoints, training metrics and loss curve.

Task 5 – RAFT Inference and Visualization

Loaded the best checkpoint from Task 4 and performed inference on Sintel and KITTI image pairs.

Generated:

Predicted optical flow arrays

Colour-coded optical flow visualizations

Flow magnitude maps

Input image pairs

Numerical inference summaries

The predicted flow fields were resized to their original image resolutions.

Task 6 – Quantitative Evaluation

Evaluated the trained model using 10 evenly spaced image pairs from each dataset.

Calculated:

Mean Endpoint Error (EPE)

Median EPE

Percentage of pixels below 1, 3 and 5 pixel EPE thresholds

KITTI-style Fl-all outlier percentage

Sintel Results

Samples evaluated: 10

Mean EPE: 13.3394 pixels

Mean of sample median EPEs: 13.4960 pixels

EPE below 1 pixel: 18.36%

EPE below 3 pixels: 52.90%

EPE below 5 pixels: 64.58%

KITTI Results

Samples evaluated: 10

Mean EPE: 39.7960 pixels

Mean of sample median EPEs: 27.3049 pixels

EPE below 1 pixel: 1.35%

EPE below 3 pixels: 13.36%

EPE below 5 pixels: 18.35%

Fl-all: 86.64%

The Sintel Fl-all-style value is a diagnostic only; the official KITTI outlier metric is meaningful for KITTI evaluation.

Key Observations

The complete simplified RAFT-style inference pipeline runs successfully on both datasets.

The model produces dense optical flow predictions at the original image resolutions.

The quantitative evaluation shows substantial prediction errors, particularly on KITTI.

Training for three CPU-based epochs was insufficient to demonstrate strong convergence.

Performance is affected by the simplified architecture, limited training duration and training configuration.

The Task 6 evaluation uses evenly spaced samples from the training datasets. It is not guaranteed to use only the held-out validation samples, so these results should be treated as diagnostic measurements rather than an independent benchmark.