
"""
STAGE 1 - TASK 3
SPARSE OPTICAL FLOW USING LUCAS-KANADE

Dataset: MPI Sintel - Clean
Sequence: alley_1
Frames: frame_0001.png and frame_0002.png

This script:
1. Loads two consecutive grayscale images.
2. Detects feature points using Shi-Tomasi corner detection.
3. Tracks feature points using pyramidal Lucas-Kanade.
4. Calculates motion vectors.
5. Visualizes tracked points and motion vectors.
6. Saves tracking results and numerical measurements.
"""

from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    PROJECT_ROOT
    / "MPI-Sintel-training_images"
    / "training"
    / "clean"
    / "alley_1"
)

FRAME_1_PATH = DATASET_DIR / "frame_0001.png"
FRAME_2_PATH = DATASET_DIR / "frame_0002.png"

RESULTS_DIR = PROJECT_ROOT / "Stage_1" / "results" / "Task 3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD IMAGES
# --------------------------------------------------

frame1 = cv2.imread(str(FRAME_1_PATH), cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread(str(FRAME_2_PATH), cv2.IMREAD_GRAYSCALE)

if frame1 is None or frame2 is None:
    raise FileNotFoundError(
        "Could not load input images. Check the dataset paths."
    )

if frame1.shape != frame2.shape:
    raise ValueError("Both input images must have the same dimensions.")

print("=" * 60)
print("STAGE 1 - TASK 3")
print("LUCAS-KANADE SPARSE OPTICAL FLOW")
print("=" * 60)

height, width = frame1.shape

print(f"\nImage dimensions: {width} x {height}")


# --------------------------------------------------
# 3. DETECT FEATURE POINTS
# --------------------------------------------------

# Shi-Tomasi corner detection parameters
feature_params = {
    "maxCorners": 300,
    "qualityLevel": 0.01,
    "minDistance": 7,
    "blockSize": 7
}

points_previous = cv2.goodFeaturesToTrack(
    frame1,
    mask=None,
    **feature_params
)

if points_previous is None:
    raise RuntimeError("No feature points detected in the first frame.")

print(f"\nFeature points detected: {len(points_previous)}")


# --------------------------------------------------
# 4. LUCAS-KANADE OPTICAL FLOW
# --------------------------------------------------

lk_params = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
}

points_next, status, error = cv2.calcOpticalFlowPyrLK(
    frame1,
    frame2,
    points_previous,
    None,
    **lk_params
)

if points_next is None or status is None:
    raise RuntimeError("Lucas-Kanade tracking failed.")

# Keep successfully tracked points
status = status.reshape(-1)

good_old = points_previous[status == 1]
good_new = points_next[status == 1]

print(f"Successfully tracked points: {len(good_new)}")
print(f"Tracking failures: {len(points_previous) - len(good_new)}")


# --------------------------------------------------
# 5. CALCULATE MOTION VECTORS
# --------------------------------------------------

old_coordinates = good_old.reshape(-1, 2)
new_coordinates = good_new.reshape(-1, 2)

displacements = new_coordinates - old_coordinates

u = displacements[:, 0]
v = displacements[:, 1]

magnitudes = np.sqrt(u**2 + v**2)

angles = np.degrees(np.arctan2(v, u))

print("\nMOTION STATISTICS")
print("-" * 45)

print(f"Tracked points : {len(good_new)}")
print(f"Mean horizontal displacement (u): {np.mean(u):.4f} pixels")
print(f"Mean vertical displacement (v):   {np.mean(v):.4f} pixels")
print(f"Mean motion magnitude:             {np.mean(magnitudes):.4f} pixels")
print(f"Maximum motion magnitude:          {np.max(magnitudes):.4f} pixels")
print(f"Minimum motion magnitude:          {np.min(magnitudes):.4f} pixels")


# --------------------------------------------------
# 6. VISUALIZE TRACKED POINTS
# --------------------------------------------------

# Create a color version of the first frame
tracking_image = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

# Draw tracked points and their trajectories
for old_point, new_point in zip(old_coordinates, new_coordinates):

    x1, y1 = np.round(old_point).astype(int)
    x2, y2 = np.round(new_point).astype(int)

    # Draw trajectory
    cv2.line(
        tracking_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        1
    )

    # Draw original feature point
    cv2.circle(
        tracking_image,
        (x1, y1),
        3,
        (0, 0, 255),
        -1
    )

    # Draw new feature point
    cv2.circle(
        tracking_image,
        (x2, y2),
        2,
        (255, 0, 0),
        -1
    )

cv2.imwrite(
    str(RESULTS_DIR / "tracked_feature_points.png"),
    tracking_image
)


# --------------------------------------------------
# 7. VISUALIZE OPTICAL FLOW VECTORS
# --------------------------------------------------

flow_image = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

for old_point, displacement in zip(old_coordinates, displacements):

    x, y = np.round(old_point).astype(int)

    end_point = np.round(
        old_point + displacement
    ).astype(int)

    # Draw motion vector
    cv2.arrowedLine(
        flow_image,
        (x, y),
        tuple(end_point),
        (0, 255, 0),
        1,
        tipLength=0.3
    )

    cv2.circle(
        flow_image,
        (x, y),
        2,
        (0, 0, 255),
        -1
    )

cv2.imwrite(
    str(RESULTS_DIR / "lucas_kanade_flow_vectors.png"),
    flow_image
)


# --------------------------------------------------
# 8. SAVE NUMERICAL RESULTS
# --------------------------------------------------

# Columns:
# x1, y1, x2, y2, u, v, magnitude, angle_degrees

tracking_results = np.column_stack((
    old_coordinates[:, 0],
    old_coordinates[:, 1],
    new_coordinates[:, 0],
    new_coordinates[:, 1],
    u,
    v,
    magnitudes,
    angles
))

np.savetxt(
    RESULTS_DIR / "optical_flow_vectors.csv",
    tracking_results,
    delimiter=",",
    header="x1,y1,x2,y2,u,v,magnitude,angle_degrees",
    comments="",
    fmt="%.6f"
)

np.save(
    RESULTS_DIR / "optical_flow_vectors.npy",
    tracking_results
)


# --------------------------------------------------
# 9. SAVE TRACKING SUMMARY
# --------------------------------------------------

summary = f"""
LUCAS-KANADE SPARSE OPTICAL FLOW - TASK 3

Dataset: MPI Sintel
Sequence: alley_1
Frames: frame_0001.png -> frame_0002.png

Image dimensions: {width} x {height}

Feature detection:
    Method: Shi-Tomasi
    Maximum corners: 300

Lucas-Kanade parameters:
    Window size: 21 x 21
    Pyramid levels: 3

Tracking:
    Detected points: {len(points_previous)}
    Successfully tracked: {len(good_new)}
    Tracking failures: {len(points_previous) - len(good_new)}

Motion statistics:
    Mean u: {np.mean(u):.6f} pixels
    Mean v: {np.mean(v):.6f} pixels
    Mean magnitude: {np.mean(magnitudes):.6f} pixels
    Maximum magnitude: {np.max(magnitudes):.6f} pixels
    Minimum magnitude: {np.min(magnitudes):.6f} pixels

Note:
    This is sparse optical flow.
    Motion is estimated only at selected feature points.
"""

(RESULTS_DIR / "tracking_summary.txt").write_text(
    summary,
    encoding="utf-8"
)


# --------------------------------------------------
# 10. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("TASK 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nResults saved in:\n{RESULTS_DIR}")

print("\nGenerated files:")
print("  - tracked_feature_points.png")
print("  - lucas_kanade_flow_vectors.png")
print("  - optical_flow_vectors.csv")
print("  - optical_flow_vectors.npy")
print("  - tracking_summary.txt")
