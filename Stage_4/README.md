Stage 4 – CNN-Based Feature Extraction and Correlation

Overview

This stage explores the CNN feature extraction and correlation mechanisms used in RAFT (Recurrent All-Pairs Field Transforms). Using KITTI image pairs and PyTorch, the implementation demonstrates feature encoding, feature matching, correlation volumes, local correlation lookup, and iterative flow refinement.

Tasks Completed

Task 1 – CNN Fundamentals

Built a basic CNN encoder to extract feature maps from a KITTI image. Visualized convolution and pooling layers to understand hierarchical feature representations.

Task 2 – CNN Feature Extraction

Extracted CNN features from two consecutive KITTI frames and compared their feature representations using feature differences and cosine similarity.

Task 3 – Feature Correlation and Matching

Computed all-pairs feature correlations and identified corresponding locations using mutual nearest-neighbour matching.

Task 3B – CNN Feature Match Validation

Validated sparse CNN feature matches against KITTI ground-truth optical flow using endpoint error (EPE) and Fl-all metrics.

Task 4 – All-Pairs Correlation Volume

Constructed a four-level correlation pyramid to represent similarities between all feature locations across two images.

Task 5 – Correlation Lookup and Local Matching

Implemented local correlation lookup around candidate feature coordinates using a neighbourhood radius of 2 across four pyramid levels.

Task 6 – Iterative Flow Refinement

Implemented a recurrent update mechanism to refine optical-flow estimates over five iterations and visualized the flow changes.

Technologies Used

Python

PyTorch

OpenCV

NumPy

Matplotlib

KITTI Optical Flow Dataset

Results

Generated feature maps, correlation matrices, feature matching visualizations, correlation pyramids, local lookup maps, and iterative flow visualizations.

Note: This stage demonstrates the main architectural components of RAFT using a simplified CNN and recurrent update implementation. The network is not trained, so the generated flow is not a trained RAFT prediction.