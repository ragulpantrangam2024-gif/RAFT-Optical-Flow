Stage 3 – Quantitative Optical Flow Evaluation

Overview

Stage 3 focuses on evaluating classical optical flow algorithms using the KITTI 2015 dataset. The objective is to understand algorithm performance through quantitative metrics, parameter experiments, motion analysis, texture analysis, occlusion analysis and multi-pair benchmarking.

Four algorithms were primarily evaluated:

Farnebäck

Horn–Schunck

TV-L1

DIS Medium

Tasks Completed

Task 1 – Optical Flow Evaluation Metrics

Implemented and studied quantitative metrics for evaluating optical flow performance.

Endpoint Error (EPE)

Mean and median EPE

Angular error

KITTI Fl-all outlier percentage

Percentage of pixels below selected error thresholds

These metrics provide a quantitative way to compare estimated optical flow against ground-truth flow.

Task 2 – KITTI Optical Flow Evaluation

Introduced the KITTI 2015 optical flow dataset and evaluated four classical algorithms against its ground-truth flow.

The evaluation included valid-pixel filtering, endpoint error, Fl-all, error distributions and runtime measurements.

Best result: DIS Medium achieved a mean EPE of 43.30 pixels and Fl-all of 61.41% on the selected image pair.

Task 3 – Parameter Sensitivity Analysis

Investigated how algorithm parameters influence optical flow accuracy and execution time.

Experiments included:

Farnebäck: Window sizes 9, 15 and 21.

Horn–Schunck: Alpha values 0.5, 1.0 and 2.0.

TV-L1: Lambda values 0.1, 0.15 and 0.2.

DIS: ULTRAFAST, FAST and MEDIUM presets.

The experiments demonstrated that parameter selection affects accuracy and runtime differently for each algorithm.

Task 4 – Motion-Based Error Analysis

Analysed optical flow performance according to ground-truth motion magnitude.

Motion was divided into three categories:

Small motion: 0–5 pixels

Medium motion: 5–20 pixels

Large motion: 20 pixels and above

DIS Medium achieved the lowest mean EPE in all three categories:

Small motion: 0.52 pixels

Medium motion: 3.05 pixels

Large motion: 59.96 pixels

The results show that large displacement remains challenging for classical optical flow algorithms.

Task 5 – Texture and Occlusion Analysis

Investigated how image texture and occluded regions influence optical flow accuracy.

The analysis included:

Low, medium and high texture classification.

Texture-based EPE and Fl-all evaluation.

Occluded and non-occluded region comparison.

Texture and occlusion visualizations.

DIS Medium achieved the lowest mean EPE across all three texture categories.

Occluded regions generally produced higher errors than non-occluded regions for all evaluated algorithms.

Task 6 – Multi-Sequence KITTI Benchmark

Extended the evaluation from a single image pair to 10 KITTI 2015 training image pairs.

Compared Farnebäck, Horn–Schunck, TV-L1 and DIS Medium using average EPE, Fl-all and runtime.

Aggregated results:

DIS Medium: 21.63 pixels mean EPE, 43.94% Fl-all, 0.131 seconds.

Farnebäck: 34.37 pixels mean EPE, 64.94% Fl-all, 0.165 seconds.

TV-L1: 41.32 pixels mean EPE, 81.97% Fl-all, 35.54 seconds.

Horn–Schunck: 46.10 pixels mean EPE, 96.96% Fl-all, 1.73 seconds.

DIS Medium achieved the best overall results across the selected 10 image pairs.

Technologies Used

Python

OpenCV

NumPy

Matplotlib

Pandas

KITTI 2015 Optical Flow Dataset

Conclusion

Stage 3 provided a quantitative understanding of classical optical flow algorithms using the KITTI 2015 dataset.

The experiments demonstrated the influence of algorithm parameters, motion magnitude, image texture and occlusion on optical flow accuracy.

Across the selected 10 KITTI image pairs, DIS Medium achieved the best overall accuracy and runtime among the four evaluated algorithms.

Stage 3 completed – Quantitative Optical Flow Evaluation.

Note: These results are based on selected KITTI 2015 training pairs and do not represent the complete official KITTI benchmark