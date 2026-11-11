
import os
import csv
import time
import struct
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE1 = os.path.join(ROOT, "Stage_1")
STAGE2 = os.path.join(ROOT, "Stage_2")
OUTPUT = os.path.join(STAGE2, "results", "Task 5")
os.makedirs(OUTPUT, exist_ok=True)

IMG_DIR = os.path.join(
    ROOT, "MPI-Sintel-training_images",
    "training", "clean", "alley_1"
)

IMAGE1 = os.path.join(IMG_DIR, "frame_0001.png")
IMAGE2 = os.path.join(IMG_DIR, "frame_0002.png")
GT_PATH = os.path.join(
    ROOT, "training", "flow", "alley_1", "frame_0001.flo"
)
MASK_PATH = os.path.join(
    ROOT, "training", "invalid", "alley_1", "frame_0001.png"
)

HS_PATH = os.path.join(
    STAGE2, "results", "Task 1", "horn_schunck_flow.npy"
)
TVL1_PATH = os.path.join(
    STAGE2, "results", "Task 3", "tvl1_flow.npy"
)

WARMUP = 1
TIMED_RUNS = 5

# ============================================================
# 2. LOAD DATA
# ============================================================

img1 = cv2.imread(IMAGE1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(IMAGE2, cv2.IMREAD_GRAYSCALE)
invalid = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None or invalid is None:
    raise FileNotFoundError("Missing Sintel images or invalid mask.")

if img1.shape != img2.shape or invalid.shape != img1.shape:
    raise ValueError("Image and mask dimensions do not match.")

H, W = img1.shape
valid = invalid == 0

def read_flo(path):
    with open(path, "rb") as f:
        magic = struct.unpack("<f", f.read(4))[0]
        if abs(magic - 202021.25) > 0.001:
            raise ValueError("Invalid Sintel .flo header.")
        w, h = struct.unpack("<ii", f.read(8))
        data = np.fromfile(f, dtype="<f4")
    if data.size != h * w * 2:
        raise ValueError("Invalid .flo data length.")
    return data.reshape(h, w, 2)

gt = read_flo(GT_PATH)

if gt.shape != (H, W, 2):
    raise ValueError("Ground truth dimensions do not match.")

def load_saved_flow(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing saved flow: {path}")
    flow = np.load(path)
    if flow.shape != (H, W, 2):
        raise ValueError(f"Unexpected flow shape: {path}")
    return flow.astype(np.float32)

print("=" * 65)
print("STAGE 2 - TASK 5: CLASSICAL OPTICAL FLOW COMPARISON")
print("=" * 65)
print(f"Image: {W} x {H}")
print(f"Valid pixels: {np.count_nonzero(valid)}")

# ============================================================
# 3. ALGORITHM FUNCTIONS
# ============================================================

def farneback():
    return cv2.calcOpticalFlowFarneback(
        img1, img2, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

def horn_schunck():
    # Same single-scale method as Stage 2 Task 1.
    a = img1.astype(np.float32) / 255.0
    b = img2.astype(np.float32) / 255.0

    ix = (
        cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
        + cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3, scale=1/8)
    ) / 2

    iy = (
        cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
        + cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    ) / 2

    it = b - a
    u = np.zeros_like(a)
    v = np.zeros_like(a)

    kernel = np.array([
        [1/12, 1/6, 1/12],
        [1/6, 0,   1/6],
        [1/12, 1/6, 1/12]
    ], dtype=np.float32)

    alpha = 1.0

    for _ in range(100):
        ua = cv2.filter2D(
            u, -1, kernel, borderType=cv2.BORDER_REFLECT
        )
        va = cv2.filter2D(
            v, -1, kernel, borderType=cv2.BORDER_REFLECT
        )
        residual = ix * ua + iy * va + it
        denominator = alpha**2 + ix**2 + iy**2
        u = ua - ix * residual / denominator
        v = va - iy * residual / denominator

    return np.stack((u, v), axis=-1)

def tvl1():
    if not hasattr(cv2, "optflow"):
        raise RuntimeError("Install opencv-contrib-python.")

    if hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        algorithm = cv2.optflow.DualTVL1OpticalFlow_create()
    else:
        algorithm = cv2.optflow.createOptFlow_DualTVL1()

    settings = [
        ("setTau", 0.25),
        ("setLambda", 0.15),
        ("setTheta", 0.3),
        ("setScalesNumber", 5),
        ("setWarpingsNumber", 5),
        ("setEpsilon", 0.01),
        ("setInnerIterations", 30),
        ("setOuterIterations", 10),
        ("setScaleStep", 0.8),
        ("setGamma", 0.0),
        ("setMedianFiltering", 5)
    ]

    for method, value in settings:
        if hasattr(algorithm, method):
            getattr(algorithm, method)(value)

    return algorithm.calc(img1, img2, None)

def dis_flow(preset):
    dis = cv2.DISOpticalFlow_create(preset)
    dis.setUseSpatialPropagation(True)
    dis.setVariationalRefinementIterations(5)
    dis.setVariationalRefinementAlpha(20.0)
    dis.setVariationalRefinementDelta(5.0)
    dis.setVariationalRefinementGamma(10.0)
    return dis.calc(img1, img2, None)

def lk_flow(points, level):
    return cv2.calcOpticalFlowPyrLK(
        img1, img2, points, None,
        winSize=(21, 21),
        maxLevel=level,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30, 0.01
        )
    )

# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(predicted, reference):
    p = np.asarray(predicted, dtype=np.float64)
    g = np.asarray(reference, dtype=np.float64)

    epe = np.linalg.norm(p - g, axis=-1)

    numerator = np.sum(p * g, axis=-1) + 1.0
    denominator = (
        np.sqrt(np.sum(p*p, axis=-1) + 1.0)
        * np.sqrt(np.sum(g*g, axis=-1) + 1.0)
    )

    angular = np.degrees(
        np.arccos(np.clip(numerator / denominator, -1, 1))
    )

    return {
        "mean_epe": float(np.mean(epe)),
        "median_epe": float(np.median(epe)),
        "mean_angular_error": float(np.mean(angular)),
        "epe_below_1_percent": float(np.mean(epe < 1) * 100),
        "epe_below_3_percent": float(np.mean(epe < 3) * 100),
        "epe_below_5_percent": float(np.mean(epe < 5) * 100),
    }

def benchmark(fn):
    # Warm-up before timing.
    for _ in range(WARMUP):
        fn()

    times = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    return float(np.median(times)), float(np.mean(times))

def sample_flow(flow, locations):
    maps_x = locations[:, 0].astype(np.float32).reshape(-1, 1)
    maps_y = locations[:, 1].astype(np.float32).reshape(-1, 1)

    u = cv2.remap(
        flow[:, :, 0], maps_x, maps_y,
        cv2.INTER_LINEAR
    ).ravel()

    v = cv2.remap(
        flow[:, :, 1], maps_x, maps_y,
        cv2.INTER_LINEAR
    ).ravel()

    return np.column_stack((u, v))

# ============================================================
# 5. COMPUTE DENSE ALGORITHMS
# ============================================================

dense_functions = {
    "Farneback": farneback,
    "Horn-Schunck": horn_schunck,
    "TV-L1": tvl1,
    "DIS ULTRAFAST": lambda: dis_flow(
        cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
    ),
    "DIS FAST": lambda: dis_flow(
        cv2.DISOPTICAL_FLOW_PRESET_FAST
    ),
    "DIS MEDIUM": lambda: dis_flow(
        cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    )
}

saved_flows = {
    "Horn-Schunck": HS_PATH,
    "TV-L1": TVL1_PATH
}

dense_flows = {}
dense_rows = []

for name, fn in dense_functions.items():
    print(f"\nEvaluating {name}...")

    if name in saved_flows:
        flow = load_saved_flow(saved_flows[name])
    else:
        flow = fn()

    if not np.all(np.isfinite(flow[valid])):
        raise ValueError(f"Non-finite flow values in {name}.")

    dense_flows[name] = flow

    metrics = calculate_metrics(flow[valid], gt[valid])
    median_time, mean_time = benchmark(fn)

    dense_rows.append({
        "algorithm": name,
        "evaluation": "valid image pixels",
        "sample_count": int(valid.sum()),
        **metrics,
        "median_runtime_seconds": median_time,
        "mean_runtime_seconds": mean_time
    })

    print(
        f"Mean EPE: {metrics['mean_epe']:.6f} px | "
        f"Median runtime: {median_time:.4f} s"
    )

# ============================================================
# 6. SPARSE LK AND SHARED-POINT EVALUATION
# ============================================================

points = cv2.goodFeaturesToTrack(
    img1,
    maxCorners=300,
    qualityLevel=0.01,
    minDistance=7,
    blockSize=7
)

if points is None:
    raise RuntimeError("No feature points detected.")

locations = points.reshape(-1, 2)

x = np.rint(locations[:, 0]).astype(int)
y = np.rint(locations[:, 1]).astype(int)

inside = (
    (x >= 0) & (x < W) &
    (y >= 0) & (y < H)
)

valid_points = np.zeros(len(locations), dtype=bool)
valid_points[inside] = valid[y[inside], x[inside]]

lk_predictions = {}
lk_statuses = {}

for name, level in [
    ("LK Single-Level", 0),
    ("LK Pyramidal", 3)
]:
    next_pts, status, _ = lk_flow(points, level)

    if next_pts is None or status is None:
        raise RuntimeError(f"Tracking failed: {name}")

    lk_predictions[name] = next_pts.reshape(-1, 2) - locations
    lk_statuses[name] = status.ravel().astype(bool)

# Use the same successfully tracked points for all algorithms.
common = (
    valid_points
    & lk_statuses["LK Single-Level"]
    & lk_statuses["LK Pyramidal"]
)

common_locations = locations[common]

if len(common_locations) == 0:
    raise RuntimeError("No common valid feature locations.")

gt_points = sample_flow(gt, common_locations)
point_rows = []

for name, flow in dense_flows.items():
    prediction = sample_flow(flow, common_locations)
    metrics = calculate_metrics(prediction, gt_points)

    point_rows.append({
        "algorithm": name,
        "evaluation": "common feature points",
        "sample_count": len(common_locations),
        **metrics
    })

for name, level in [
    ("LK Single-Level", 0),
    ("LK Pyramidal", 3)
]:
    prediction = lk_predictions[name][common]
    metrics = calculate_metrics(prediction, gt_points)

    median_time, mean_time = benchmark(
        lambda level=level: lk_flow(points, level)
    )

    point_rows.append({
        "algorithm": name,
        "evaluation": "common feature points",
        "sample_count": len(common_locations),
        **metrics
    })

    # Save LK timing separately, not in dense table.
    print(
        f"{name}: mean EPE={metrics['mean_epe']:.6f}, "
        f"median runtime={median_time:.4f} s"
    )

    point_rows[-1]["median_runtime_seconds"] = median_time
    point_rows[-1]["mean_runtime_seconds"] = mean_time

# ============================================================
# 7. SAVE CSV FILES
# ============================================================

def save_csv(filename, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(
        os.path.join(OUTPUT, filename),
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

save_csv("dense_comparison_metrics.csv", dense_rows)
save_csv("shared_feature_comparison_metrics.csv", point_rows)

# ============================================================
# 8. ACCURACY AND RUNTIME PLOTS
# ============================================================

names = [r["algorithm"] for r in dense_rows]
epe_values = [r["mean_epe"] for r in dense_rows]
times = [r["median_runtime_seconds"] for r in dense_rows]

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(names, epe_values)
ax.set_ylabel("Mean EPE (pixels)")
ax.set_title("Dense Optical Flow - Accuracy Comparison")
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "dense_accuracy_comparison.png"),
    dpi=200
)
plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(names, times)
ax.set_ylabel("Median runtime (seconds)")
ax.set_title("Dense Optical Flow - Runtime Comparison")
ax.tick_params(axis="x", rotation=25)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "dense_runtime_comparison.png"),
    dpi=200
)
plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(
    [r["algorithm"] for r in point_rows],
    [r["mean_epe"] for r in point_rows]
)
ax.set_ylabel("Mean EPE (pixels)")
ax.set_title(
    f"Common Feature Locations ({len(common_locations)} points)"
)
ax.tick_params(axis="x", rotation=30)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "shared_feature_accuracy.png"),
    dpi=200
)
plt.close()

# ============================================================
# 9. SIDE-BY-SIDE DENSE FLOW VISUALIZATION
# ============================================================

def flow_to_color(flow, max_magnitude):
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag, angle = cv2.cartToPolar(u, v, angleInDegrees=True)

    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.uint8(angle / 2)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.uint8(
        np.clip(mag / max_magnitude * 255, 0, 255)
    )

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

# Shared color scaling is essential for fair visual comparison.
all_magnitudes = [
    np.linalg.norm(flow, axis=2)
    for flow in dense_flows.values()
]
max_mag = max(
    float(np.percentile(mag[valid], 99))
    for mag in all_magnitudes
)
max_mag = max(max_mag, 1e-6)

fig, axes = plt.subplots(2, 3, figsize=(17, 8))

for ax, (name, flow) in zip(axes.flat, dense_flows.items()):
    ax.imshow(flow_to_color(flow, max_mag))
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT, "dense_flow_visual_comparison.png"),
    dpi=180
)
plt.close()

# ============================================================
# 10. SAVE SUMMARY
# ============================================================

best_accuracy = min(
    dense_rows, key=lambda row: row["mean_epe"]
)
fastest = min(
    dense_rows, key=lambda row: row["median_runtime_seconds"]
)

summary_path = os.path.join(OUTPUT, "comparison_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("STAGE 2 - TASK 5\n")
    f.write("CLASSICAL OPTICAL FLOW COMPARISON\n\n")
    f.write("Dataset: MPI Sintel, alley_1\n")
    f.write("Frames: frame_0001 -> frame_0002\n")
    f.write(f"Resolution: {W} x {H}\n")
    f.write(f"Valid pixels: {int(valid.sum())}\n")
    f.write(f"Common feature points: {int(common.sum())}\n")
    f.write(f"Timed runs: {TIMED_RUNS}\n\n")

    f.write("DENSE ALGORITHMS\n")
    for row in dense_rows:
        f.write(
            f"{row['algorithm']}: "
            f"EPE={row['mean_epe']:.6f} px, "
            f"angular={row['mean_angular_error']:.6f} deg, "
            f"runtime={row['median_runtime_seconds']:.6f} s\n"
        )

    f.write("\nCOMMON FEATURE-POINT COMPARISON\n")
    for row in point_rows:
        f.write(
            f"{row['algorithm']}: "
            f"EPE={row['mean_epe']:.6f} px\n"
        )

    f.write("\nBEST DENSE ACCURACY: ")
    f.write(best_accuracy["algorithm"] + "\n")
    f.write("FASTEST DENSE ALGORITHM: ")
    f.write(fastest["algorithm"] + "\n")

    f.write(
        "\nLimitations: One Sintel frame pair. "
        "Runtime includes algorithm initialization for "
        "DIS and TV-L1. Saved Horn-Schunck and TV-L1 "
        "flows are used for accuracy evaluation. "
        "This is not an official Sintel benchmark.\n"
    )

print("\n" + "=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print("Best dense accuracy:", best_accuracy["algorithm"])
print("Fastest dense method:", fastest["algorithm"])
print("Common feature points:", len(common_locations))
print("Results saved to:", OUTPUT)
print("\nStage 2 Task 5 completed.")
