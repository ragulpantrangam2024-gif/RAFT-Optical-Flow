Stage 1 – Classical Optical Flow Fundamentals

Overview

Stage 1 establishes the mathematical and practical foundations of optical flow using classical computer vision techniques.

Dataset: MPI Sintel – Clean Pass
Sequence: alley_1
Frame pair: frame_0001.png → frame_0002.png
Image resolution: 1024 × 436
Tools: Python, OpenCV, NumPy, Matplotlib

Tasks

Task

Description

Technique

1

Image differences and motion visualization

Absolute pixel difference

2

Brightness constancy and optical flow constraint

Image derivatives ((I_x, I_y, I_t))

3

Sparse optical flow estimation

Lucas–Kanade

4

Dense optical flow estimation

Farnebäck

5

Optical flow evaluation against ground truth

Endpoint Error (EPE), Angular Error

6

Comparative evaluation of sparse and dense flow

Lucas–Kanade vs Farnebäck

7

Optical flow limitations and failure analysis

Motion, texture and occlusion-mask analysis

Key Concepts

Brightness Constancy: Assumes that the intensity of a moving pixel remains approximately constant between consecutive frames.

Optical Flow Constraint: (I_xu + I_yv + I_t = 0)

Lucas–Kanade: Estimates motion at selected feature points.

Farnebäck: Estimates dense motion across the image.

Endpoint Error: Measures the distance between estimated and ground-truth flow vectors.

Failure Analysis: Investigates the influence of image texture, motion magnitude and occlusion-mask groups on estimation accuracy.

Experimental Results

Task 2 – Image Derivatives

Derivative

Standard Deviation

Horizontal ((I_x))

0.032161

Vertical ((I_y))

0.034739

Temporal ((I_t))

0.080231

Task 4 – Farnebäck Dense Optical Flow

Measurement

Result

Mean flow magnitude

2.382 px

Median flow magnitude

1.490 px

Maximum flow magnitude

8.595 px

Task 5 – Ground-Truth Evaluation

Metric

Result

Mean EPE

0.868818 px

Median EPE

0.112741 px

Mean Angular Error

6.160739°

EPE below 1 px

78.90%

EPE below 3 px

88.40%

EPE below 5 px

93.76%

Task 6 – Lucas–Kanade vs Farnebäck

Both algorithms were evaluated at the same 300 feature locations.

Metric

Lucas–Kanade

Farnebäck

Mean EPE

0.403739 px

0.400230 px

Median EPE

0.036211 px

0.034445 px

Mean Angular Error

2.753454°

2.810405°

Task 7 – Failure Analysis

Analysis

Mean EPE

Low-texture regions

1.7413 px

Medium-low texture

0.8633 px

Medium-high texture

0.4826 px

High-texture regions

0.3889 px

Ground-truth motion 5–10 px

2.1280 px

Pixels with occlusion-mask value 255 had a mean EPE of 2.4606 px, compared with 0.8363 px for mask value 0.

Results Structure

Stage_1/
│
├── task_01_optical_flow_motion_fields.py
├── task_02_brightness_constancy_optical_flow_constraint.py
├── task_03_lucas_kanade_sparse_optical_flow.py
├── task_04_farneback_dense_optical_flow.py
├── task_05_optical_flow_evaluation.py
├── task_06_lucas_kanade_vs_farneback.py
├── task_07_optical_flow_failure_analysis.py
│
└── results/
    ├── Task 1/
    ├── Task 2/
    ├── Task 3/
    ├── Task 4/
    ├── Task 5/
    ├── Task 6/
    └── Task 7/

Key Observations

Lucas–Kanade provides sparse motion estimates, whereas Farnebäck provides dense motion estimates.

Both methods achieved similar EPE at the same selected feature locations in this experiment.

Low-texture regions showed higher optical flow errors.

Larger ground-truth motion was associated with higher estimation error in the evaluated motion groups.

Errors were higher for pixels with occlusion-mask value 255.

Limitations

Experiments primarily use one MPI Sintel sequence and one frame pair.

Results are specific to the selected scene and algorithm parameters.

The reported metrics should not be interpreted as full-dataset benchmark results.

Conclusion

Stage 1 establishes the fundamentals of classical optical flow, including mathematical formulation, sparse and dense estimation, ground-truth evaluation, algorithm comparison and failure analysis.

Next Stage: Deep Learning-Based Optical Flow and RAFT (Recurrent All-Pairs Field Transforms).