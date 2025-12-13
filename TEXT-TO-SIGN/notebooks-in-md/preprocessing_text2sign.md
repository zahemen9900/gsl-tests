# Text2Sign — Local Preprocessing Notebook (Markdown to copy into .ipynb)

> Adapted from `signtalk_local_preprocessing.md`, tailored for generative Text→Sign. Preserves motion diversity (less aggressive filtering), uses world coordinates when available, normalizes into a canonical avatar space, and exports mean/std + sequence-length stats for training.

---

## Cell 0 — Dependencies (run once)

```bash
pip install mediapipe opencv-python-headless pandas numpy tqdm scipy
```

---

## Cell 1 — Imports & Setup

```python
import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
from scipy import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("✅ Imports successful")
```

---

## Cell 2 — Paths, Feature Shapes, Thresholds

```python
DATASET_DIR = "./text2sign_dataset"  # update if different
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
META_PATH = os.path.join(DATASET_DIR, "metadata.csv")

OUTPUT_DIR = "./features/text2sign_pose"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GLOBAL_STATS_PATH = "./proc/text2sign_global_stats.npz"
os.makedirs(os.path.dirname(GLOBAL_STATS_PATH), exist_ok=True)

# Feature dimensions (world coords preferred)
FACE_DOWNSAMPLE = 5
NUM_FACE_LANDMARKS = len(range(0, 468, FACE_DOWNSAMPLE))  # 94
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

POSE_FEATURES = NUM_POSE_LANDMARKS * 4          # x,y,z,visibility
HANDS_FEATURES = 2 * NUM_HAND_LANDMARKS * 3      # both hands, xyz
FACE_FEATURES = NUM_FACE_LANDMARKS * 3
PER_FRAME_FEATURES = POSE_FEATURES + HANDS_FEATURES + FACE_FEATURES

# Quality / processing knobs
TARGET_SEQ_LEN = 96          # generative: keep longer motion when present
TARGET_FPS = 30              # resample to fixed FPS
MODEL_COMPLEXITY = 2
MIN_FRAMES = 32              # lighter gate vs classification
MIN_VALID_RATIO = 0.30       # allow more sparse detections
EMA_ALPHA = 0.15             # light temporal smoothing
VISIBILITY_THRESHOLD = 0.35  # keep more frames
MOTION_KEEP_THRESHOLD = 5e-4 # lenient; we want pauses too
MOTION_REJECTION_THRESHOLD = 0.0  # disable hard rejection for generation
KEEP_LOW_QUALITY = True      # keep flagged clips, mark in metadata

LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12; LEFT_HIP = 23; RIGHT_HIP = 24

print("📐 Feature dimensions:")
print(f"  • Per-frame: {PER_FRAME_FEATURES}")
print("⚙️ Quality (lenient for generation):")
print(f"  • min_frames: {MIN_FRAMES}, min_valid_ratio: {MIN_VALID_RATIO}")
print(f"  • visibility min: {VISIBILITY_THRESHOLD}, motion keep ≥ {MOTION_KEEP_THRESHOLD}")
print(f"  • Target FPS: {TARGET_FPS}, Target seq len: {TARGET_SEQ_LEN}")
```

---

## Cell 3 — Metadata Loader (robust CSV)

```python
import csv
records = []
with open(META_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

df_meta = pd.DataFrame(records)
df_meta['sentence_id'] = df_meta['sentence_id'].astype(str)

print(f"✅ Loaded metadata: {len(df_meta)} videos")
print(df_meta.head())
```

---

## Cell 4 — Geometry Helpers (Canonical Avatar)

```python
mp_holistic = mp.solutions.holistic
HOLISTIC = None
FACE_INDICES = list(range(0, 468, FACE_DOWNSAMPLE))

# --- Canonicalization steps ---
def canonicalize(points_xyz: np.ndarray):
    """points_xyz: (N,3) in world or image space. Returns centered + scaled + aligned."""
    pts = points_xyz.copy()
    root = 0.5 * (pts[LEFT_HIP] + pts[RIGHT_HIP])
    pts -= root

    spine_vec = 0.5 * (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) - 0.5 * (pts[LEFT_HIP] + pts[RIGHT_HIP])
    spine_len = np.linalg.norm(spine_vec)
    if spine_len < 1e-4:
        spine_len = 1.0
    pts /= spine_len

    shoulders = pts[RIGHT_SHOULDER] - pts[LEFT_SHOULDER]
    yaw = np.arctan2(shoulders[2], shoulders[0] + 1e-8)
    rot = np.array([[ np.cos(-yaw), 0, np.sin(-yaw)],
                    [ 0,            1, 0           ],
                    [-np.sin(-yaw), 0, np.cos(-yaw)]], dtype=np.float32)
    pts = pts @ rot.T
    return pts.astype(np.float32)


def trim_or_pad_sequence(sequence: np.ndarray, target_len: int = TARGET_SEQ_LEN) -> np.ndarray:
    frames = sequence.shape[0]
    if frames == target_len:
        return sequence
    if frames > target_len:
        start = (frames - target_len) // 2
        return sequence[start:start + target_len]
    pad_len = target_len - frames
    pad = np.zeros((pad_len, sequence.shape[1]), dtype=sequence.dtype)
    return np.vstack([sequence, pad])


def exponential_smooth(sequence: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    if alpha <= 0.0 or sequence.shape[0] < 2:
        return sequence.astype(np.float32)
    out = sequence.astype(np.float32, copy=True)
    one_minus = 1.0 - alpha
    for i in range(1, out.shape[0]):
        out[i] = alpha * out[i] + one_minus * out[i - 1]
    return out


def resample_frames(frames: np.ndarray, orig_fps: float, target_fps: float = TARGET_FPS):
    if orig_fps <= 0 or abs(orig_fps - target_fps) < 1e-3:
        return frames
    t_orig = np.linspace(0, frames.shape[0] - 1, frames.shape[0])
    t_new = np.linspace(0, frames.shape[0] - 1, int(frames.shape[0] * target_fps / max(orig_fps, 1e-6)))
    resampled = np.stack([np.interp(t_new, t_orig, frames[:, i]) for i in range(frames.shape[1])], axis=1)
    return resampled.astype(np.float32)


def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    diffs = np.diff(frames, axis=0, prepend=frames[:1])
    energy = np.linalg.norm(diffs, axis=1)
    return energy.astype(np.float32)


def summarize_motion(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }

print("✅ Helpers ready (canonicalization, smoothing, resampling)")
```

---

## Cell 5 — MediaPipe Worker Init

```python
from functools import partial
from multiprocessing import Pool, cpu_count
import atexit


def worker_init(face_downsample: int = FACE_DOWNSAMPLE):
    global HOLISTIC, FACE_INDICES
    try:
        HOLISTIC = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        HOLISTIC = None
    FACE_INDICES = list(range(0, 468, max(1, int(face_downsample))))


def _close_holistic():
    global HOLISTIC
    try:
        if HOLISTIC is not None:
            HOLISTIC.close()
    except Exception:
        pass

atexit.register(_close_holistic)
print("✅ Worker initializer ready")
```

---

## Cell 6 — Single-Video Processing (world coords + lenient gating)

```python
def process_single_video(row_dict, video_dir, output_dir, face_downsample):
    import os
    video_file = row_dict['video_file']
    video_path = os.path.join(video_dir, video_file)
    if not os.path.exists(video_path):
        return {'status': 'missing', 'video_file': video_file, 'error': 'File not found'}

    cap = None
    local_holistic = None
    holistic_inst = HOLISTIC
    try:
        if holistic_inst is None:
            local_holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=MODEL_COMPLEXITY,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            holistic_inst = local_holistic

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'status': 'failed', 'video_file': video_file, 'error': 'Cannot open video'}
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS

        frames = []
        quality_flags = []
        vis_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic_inst.process(rgb)

            # Use world landmarks when available; fallback to image landmarks
            pose_lm = res.pose_world_landmarks or res.pose_landmarks
            if pose_lm is None:
                pose_block = [0.0] * (NUM_POSE_LANDMARKS * 4)
                pose_vis = 0.0
            else:
                pose_block = []
                for lm in pose_lm.landmark:
                    pose_block.extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)])
                pose_vis = float(np.mean([getattr(lm, 'visibility', 1.0) for lm in pose_lm.landmark]))

            if res.left_hand_landmarks:
                lh = [v for lm in res.left_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
            else:
                lh = [0.0] * (NUM_HAND_LANDMARKS * 3)

            if res.right_hand_landmarks:
                rh = [v for lm in res.right_hand_landmarks.landmark for v in (lm.x, lm.y, lm.z)]
            else:
                rh = [0.0] * (NUM_HAND_LANDMARKS * 3)

            if res.face_landmarks:
                fl = res.face_landmarks.landmark
                face_block = []
                for i in range(0, 468, face_downsample):
                    lm = fl[i]
                    face_block.extend([lm.x, lm.y, lm.z])
            else:
                face_block = [0.0] * (NUM_FACE_LANDMARKS * 3)

            frame_vec = pose_block + lh + rh + face_block
            if len(frame_vec) < PER_FRAME_FEATURES:
                frame_vec.extend([0.0] * (PER_FRAME_FEATURES - len(frame_vec)))
            frame_arr = np.asarray(frame_vec[:PER_FRAME_FEATURES], dtype=np.float32)

            pose_xyz = frame_arr[:POSE_FEATURES].reshape(NUM_POSE_LANDMARKS, 4)
            coords = pose_xyz[:, :3]
            coords = canonicalize(coords)
            pose_xyz[:, :3] = coords
            frame_arr[:POSE_FEATURES] = pose_xyz.reshape(-1)

            frames.append(frame_arr)
            quality_flags.append(pose_lm is not None or res.left_hand_landmarks or res.right_hand_landmarks)
            vis_scores.append(pose_vis)

        cap.release()

        if not frames:
            return {'status': 'failed', 'video_file': video_file, 'error': 'No frames'}

        frames = np.stack(frames, axis=0)
        motion_vals = compute_motion_energy(frames)
        quality_mask = np.array(quality_flags, dtype=bool)
        vis_scores = np.array(vis_scores, dtype=np.float32)

        valid_idx = np.where(quality_mask)[0]
        detection_ratio = float(valid_idx.size / len(frames)) if len(frames) else 0.0
        if valid_idx.size == 0:
            return {'status': 'failed', 'video_file': video_file, 'error': 'No valid detections'}

        frames = frames[valid_idx]
        vis_scores = vis_scores[valid_idx]
        motion_vals = motion_vals[valid_idx]

        keep_mask = (vis_scores >= VISIBILITY_THRESHOLD) | (motion_vals >= MOTION_KEEP_THRESHOLD)
        if not keep_mask.any():
            keep_mask[0] = True
        frames = frames[keep_mask]
        vis_scores = vis_scores[keep_mask]
        motion_vals = motion_vals[keep_mask]

        # Resample to fixed FPS (lenient; keeps pauses)
        frames = resample_frames(frames, orig_fps, TARGET_FPS)

        # Light smoothing
        frames = exponential_smooth(frames, alpha=EMA_ALPHA)

        # Optional: trim/pad for downstream batching
        frames_tp = trim_or_pad_sequence(frames, TARGET_SEQ_LEN)

        motion_summary = summarize_motion(motion_vals)
        mean_vis = float(vis_scores.mean()) if vis_scores.size else 0.0

        status = 'success'
        if (frames.shape[0] < MIN_FRAMES or detection_ratio < MIN_VALID_RATIO) and not KEEP_LOW_QUALITY:
            status = 'low_quality'
        elif (frames.shape[0] < MIN_FRAMES or detection_ratio < MIN_VALID_RATIO) and KEEP_LOW_QUALITY:
            status = 'low_quality_kept'

        base = os.path.splitext(video_file)[0]
        npy_path = os.path.join(output_dir, f"{base}.npy")
        np.save(npy_path, frames.astype(np.float32))

        payload = {
            'video_file': video_file,
            'feature_path': npy_path,
            'num_frames': int(frames.shape[0]),
            'frame_feature_dim': int(frames.shape[1]),
            'detection_ratio': detection_ratio,
            'mean_visibility': mean_vis,
            'motion_mean': float(motion_summary.get('mean', 0.0)),
            'motion_p50': float(motion_summary.get('p50', 0.0)),
            'motion_p95': float(motion_summary.get('p95', 0.0)),
            'motion_max': float(motion_summary.get('max', 0.0)),
            'sentence_id': row_dict.get('sentence_id'),
            'sentence_text': row_dict.get('sentence'),
            'sentence_gloss': row_dict.get('gloss', ''),
            'orig_fps': float(orig_fps),
            'status': status,
        }
        return payload

    except Exception as exc:
        return {'status': 'failed', 'video_file': video_file, 'error': str(exc)}

    finally:
        try:
            if cap is not None:
                cap.release()
            if local_holistic is not None:
                local_holistic.close()
        except Exception:
            pass
```

---

## Cell 7 — Parallel Execution

```python
num_workers = max(1, cpu_count() - 2)
print(f"🚀 Starting extraction | workers={num_workers} | videos={len(df_meta)}")

video_records = df_meta.to_dict('records')
process_func = partial(process_single_video, video_dir=VIDEO_DIR, output_dir=OUTPUT_DIR, face_downsample=FACE_DOWNSAMPLE)

results = []
with Pool(num_workers, initializer=worker_init, initargs=(FACE_DOWNSAMPLE,)) as pool:
    for res in tqdm(pool.imap(process_func, video_records), total=len(video_records), desc="Processing"):
        results.append(res)

processed_records = []
failed = []
missing = []
lowq = []
for r in results:
    status = r.get('status')
    if status == 'success' or status == 'low_quality_kept':
        processed_records.append(r)
        if status == 'low_quality_kept':
            lowq.append({**r, 'kept': True})
    elif status == 'low_quality':
        lowq.append({**r, 'kept': False})
    elif status == 'failed':
        failed.append(r)
    elif status == 'missing':
        missing.append(r)

print(f"✅ Processed: {len(processed_records)} | ⚠️ LowQ: {len(lowq)} | ❌ Failed: {len(failed)} | 🚫 Missing: {len(missing)}")
```

---

## Cell 8 — Persist Metadata & Debug

```python
df_processed = pd.DataFrame(processed_records)
output_meta_path = "proc/text2sign_processed_metadata.csv"
df_processed.to_csv(output_meta_path, index=False)
print(f"✅ Saved processed metadata: {output_meta_path}")

if failed:
    pd.DataFrame(failed).to_csv("proc/text2sign_failed_videos.csv", index=False)
if missing:
    pd.DataFrame(missing).to_csv("proc/text2sign_missing_videos.csv", index=False)
if lowq:
    pd.DataFrame(lowq).to_csv("proc/text2sign_low_quality.csv", index=False)
```

---

## Cell 9 — Global Stats (mean/std + lengths)

```python
if not df_processed.empty:
    total_frames = 0
    sum_feats = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
    sum_sq = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
    seq_lengths = []
    motion_means = []

    for row in tqdm(df_processed.itertuples(), total=len(df_processed), desc="Stats"):
        arr = np.load(row.feature_path).astype(np.float32)
        total_frames += arr.shape[0]
        sum_feats += arr.sum(axis=0)
        sum_sq += (arr ** 2).sum(axis=0)
        seq_lengths.append(arr.shape[0])
        if hasattr(row, 'motion_mean'):
            motion_means.append(row.motion_mean)

    feature_mean = sum_feats / max(total_frames, 1)
    feature_var = sum_sq / max(total_frames, 1) - feature_mean ** 2
    feature_std = np.sqrt(np.maximum(feature_var, 1e-8))

    np.savez(
        GLOBAL_STATS_PATH,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        seq_lengths=np.array(seq_lengths, dtype=np.int32),
        motion_means=np.array(motion_means, dtype=np.float32),
    )

    print(f"✅ Saved global stats → {GLOBAL_STATS_PATH}")
    print(f"   • Avg seq len: {np.mean(seq_lengths):.1f}")
    print(f"   • 95p seq len: {np.percentile(seq_lengths, 95):.1f}")
```

---

## Cell 10 — Quick Summary

```python
print("=" * 50)
print("📊 TEXT2SIGN PREPROCESSING SUMMARY")
print("=" * 50)
print(f"Videos processed: {len(df_processed)}")
print(f"Features dir: {OUTPUT_DIR}")
print(f"Metadata: proc/text2sign_processed_metadata.csv")
if os.path.exists(GLOBAL_STATS_PATH):
    stats = np.load(GLOBAL_STATS_PATH)
    print(f"Mean/std saved with dim: {stats['feature_mean'].shape[0]}")
print(f"Median frames: {df_processed['num_frames'].median() if not df_processed.empty else 0}")
print(f"Median detection ratio: {df_processed['detection_ratio'].median() if 'detection_ratio' in df_processed else 0}")
```

---

## Cell 11 — Visualization: Frame & Category Distributions

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Frame count distribution
plt.subplot(1, 2, 1)
df_processed['num_frames'].hist(bins=30, edgecolor='black')
plt.xlabel('Number of Frames')
plt.ylabel('Frequency')
plt.title('Distribution of Frame Counts')
plt.axvline(df_processed['num_frames'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()

# Category distribution
plt.subplot(1, 2, 2)
category_counts = df_processed['category'].value_counts()
category_counts.plot(kind='barh')
plt.xlabel('Number of Videos')
plt.title('Videos per Category')
plt.tight_layout()
plt.show()
```

---

## Cell 12 — Visualization: Sample Feature Slice

```python
import matplotlib.pyplot as plt

sample_row = df_processed.sample(1).iloc[0]
arr = np.load(sample_row['feature_path'])

print(f"📹 Video: {sample_row['video_file']}")
print(f"📝 Sentence: {sample_row.get('sentence_text', sample_row.get('sentence', ''))}")
print(f"🎬 Frames: {sample_row['num_frames']}")
print(f"📈 Detection ratio: {sample_row.get('detection_ratio', 0):.2f}")

# Plot first joint x trajectory (pose joint 0 x)
pose_x = arr[:, 0]
plt.figure(figsize=(10, 3))
plt.plot(pose_x)
plt.title('Pose Joint 0 - X over time')
plt.xlabel('Frame')
plt.ylabel('Value')
plt.tight_layout()
plt.show()
```

