Stage 2 – Advanced Classical Optical Flow Algorithms

Overview

This stage focuses on implementing and understanding advanced classical optical flow algorithms using Python, OpenCV and NumPy.

The main objective is to explore different approaches to motion estimation, evaluate their accuracy using ground-truth optical flow, and compare their computational performance.

Dataset: MPI Sintel
Sequence: alley_1
Frames: frame_0001.png → frame_0002.png
Resolution: 1024 × 436

Tasks

Task 1 – Horn–Schunck Optical Flow

Implemented dense optical flow from scratch using NumPy and OpenCV.

Implemented global smoothness regularization.

Calculated horizontal and vertical flow components.

Visualized flow direction, motion magnitude and flow vectors.

Used a smoothness parameter (alpha) of 1.0 and 100 iterations.

Mean EPE: 2.713666 pixels (evaluated in Task 5).

Task 2 – Pyramidal Lucas–Kanade Optical Flow

Implemented pyramidal Lucas–Kanade tracking using Shi–Tomasi feature detection.

Compared single-level and pyramidal Lucas–Kanade.

Used image pyramids to estimate motion at multiple resolutions.

Tracked 300 feature points.

Evaluated tracking accuracy using Sintel ground truth.

Results:

Single-level LK mean EPE: 0.575360 pixels.

Pyramidal LK mean EPE: 0.403739 pixels.

Mean EPE improved by approximately 29.8% using pyramidal processing.

Task 3 – TV-L1 Optical Flow

Implemented dense optical flow using the TV-L1 algorithm available in OpenCV Contrib.

Explored L1 data-term optimization.

Studied total variation regularization.

Generated dense flow visualizations and motion magnitude maps.

Evaluated results against Sintel ground truth.

Results:

Mean EPE: 0.371434 pixels.

Median EPE: 0.052297 pixels.

Mean angular error: 4.237141 degrees.

EPE below 1 pixel: 92.96%.

Task 4 – DIS Optical Flow

Implemented Dense Inverse Search (DIS) optical flow using OpenCV.

Evaluated three DIS configurations:

ULTRAFAST

FAST

MEDIUM

Compared their accuracy and computational runtime.

Results:

DIS ULTRAFAST mean EPE: 0.460078 pixels.

DIS FAST mean EPE: 0.460850 pixels.

DIS MEDIUM mean EPE: 0.292353 pixels.

DIS MEDIUM achieved the best accuracy among the three presets in this experiment, while FAST provided the shortest runtime in the initial preset comparison.

Task 5 – Classical Optical Flow Algorithm Comparison

Conducted a comparative evaluation of classical optical flow algorithms using the same MPI Sintel frame pair.

Dense optical flow results:

Farnebäck – Mean EPE: 0.868818 pixels.

Horn–Schunck – Mean EPE: 2.713666 pixels.

TV-L1 – Mean EPE: 0.371434 pixels.

DIS ULTRAFAST – Mean EPE: 0.460078 pixels.

DIS FAST – Mean EPE: 0.460850 pixels.

DIS MEDIUM – Mean EPE: 0.292353 pixels.

Sparse optical flow results:

Single-level Lucas–Kanade – Mean EPE: 0.575360 pixels.

Pyramidal Lucas–Kanade – Mean EPE: 0.403739 pixels.

Also compared computational runtime and generated accuracy and runtime visualizations.

Key findings:

DIS MEDIUM achieved the best dense-flow accuracy in this experiment.

DIS ULTRAFAST provided the fastest dense-flow execution in the final comparison.

Pyramidal Lucas–Kanade improved tracking accuracy over single-level Lucas–Kanade.

TV-L1 achieved good accuracy but required considerably more processing time.

Different algorithms offer different trade-offs between accuracy and computational efficiency.

Technologies Used

Python

OpenCV

OpenCV Contrib

NumPy

Matplotlib

MPI Sintel Dataset

Project Structure

Stage_2/
│
├── task_01_horn_schunck_optical_flow.py
├── task_02_pyramidal_lucas_kanade.py
├── task_03_tvl1_optical_flow.py
├── task_04_dis_optical_flow.py
├── task_05_classical_algorithm_comparison.py
│
├── results/
│   ├── Task 1/
│   ├── Task 2/
│   ├── Task 3/
│   ├── Task 4/
│   └── Task 5/
│
└── README.md

Limitations

Experiments were primarily conducted using one MPI Sintel sequence and one frame pair.

Results do not represent the complete Sintel benchmark.

Runtime measurements depend on hardware and execution conditions.

Sparse and dense optical flow methods were evaluated separately because they use different evaluation regions.

Conclusion

Stage 2 explored several classical optical flow approaches, including global smoothness-based estimation, sparse feature tracking, optimization-based dense flow and efficient patch-based methods.

The experiments provided practical understanding of optical flow estimation, ground-truth evaluation, motion visualization, accuracy and computational performance.