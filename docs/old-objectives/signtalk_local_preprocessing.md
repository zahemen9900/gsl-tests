# 💻 SignTalk-GH: Local MediaPipe Preprocessing (Motion-Gated Pipeline)

This notebook now mirrors the motion-aware pipeline used during training. Key updates:

- **Motion + visibility gating** removes idle clips before saving features.
- **EMA smoothing and torso normalization** stabilize landmark coordinates.
- **Motion diagnostics** (mean, p50, p95, max) are persisted for audits.
- **Global statistics export** (`proc/global_stats.npz`) keeps training normalization aligned.

Run the notebook cells top-to-bottom with the reference below.

---

## Cell 0 — Dependencies (run once)

```bash
pip install mediapipe opencv-python pandas numpy tqdm
```

Use `opencv-python-headless` if the GUI build fails.

---

## Cell 1 — Imports & Setup

```python
import os
import re
import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("✅ Imports successful")
```

---

## Cell 2 — Paths, Feature Shapes, Thresholds

```python
DATASET_DIR = "./SignTalk-GH_Sampled"
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
META_PATH = os.path.join(DATASET_DIR, "sampled_metadata.csv")

OUTPUT_DIR = "./features/pose_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GLOBAL_STATS_PATH = "./proc/global_stats.npz"
os.makedirs(os.path.dirname(GLOBAL_STATS_PATH), exist_ok=True)

FACE_DOWNSAMPLE = 5
NUM_FACE_LANDMARKS = len(range(0, 468, FACE_DOWNSAMPLE))
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

POSE_FEATURES = NUM_POSE_LANDMARKS * 4
HANDS_FEATURES = 2 * NUM_HAND_LANDMARKS * 3
FACE_FEATURES = NUM_FACE_LANDMARKS * 3
PER_FRAME_FEATURES = POSE_FEATURES + HANDS_FEATURES + FACE_FEATURES

TARGET_SEQ_LEN = 64
MODEL_COMPLEXITY = 2
MIN_FRAMES = 48
MIN_VALID_RATIO = 0.45
EMA_ALPHA = 0.2
VISIBILITY_THRESHOLD = 0.55
MOTION_KEEP_THRESHOLD = 1e-3
MOTION_REJECTION_THRESHOLD = 7.5e-4
KEEP_LOW_QUALITY = False

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

print("📐 Per-frame features:")
print(f"  Pose: {POSE_FEATURES}")
print(f"  Hands: {HANDS_FEATURES}")
print(f"  Face: {FACE_FEATURES}")
print(f"  Total: {PER_FRAME_FEATURES}")
print("⚙️  Quality thresholds configured")
```

---

## Cell 3 — Robust Metadata Loader

```python
records = []
with open(META_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

df_meta = pd.DataFrame(records)
df_meta['sentence_id'] = df_meta['sentence_id'].astype(int)

print(f"✅ Loaded metadata: {len(df_meta)} videos")
print(df_meta.head())
```

---

## Cell 4 — Feature Extraction Helpers

```python
mp_holistic = mp.solutions.holistic

def normalize_frame_landmarks(frame_feats):
    feats = np.asarray(frame_feats, dtype=np.float32)

    pose = feats[:POSE_FEATURES].reshape(NUM_POSE_LANDMARKS, 4)
    left_hand_start = POSE_FEATURES
    left_hand_end = left_hand_start + NUM_HAND_LANDMARKS * 3
    right_hand_end = left_hand_end + NUM_HAND_LANDMARKS * 3
    left_hand = feats[left_hand_start:left_hand_end].reshape(NUM_HAND_LANDMARKS, 3)
    right_hand = feats[left_hand_end:right_hand_end].reshape(NUM_HAND_LANDMARKS, 3)
    face = feats[right_hand_end:right_hand_end + NUM_FACE_LANDMARKS * 3].reshape(NUM_FACE_LANDMARKS, 3)

    pose_coords = pose[:, :3].copy()
    torso_pts = pose_coords[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]

    if not np.isfinite(torso_pts).all() or np.linalg.norm(torso_pts) < 1e-6:
        return feats

    torso_center = torso_pts.mean(axis=0)
    pose_coords -= torso_center
    left_hand -= torso_center
    right_hand -= torso_center
    face -= torso_center

    shoulder_span = np.linalg.norm(pose_coords[LEFT_SHOULDER] - pose_coords[RIGHT_SHOULDER])
    hip_span = np.linalg.norm(pose_coords[LEFT_HIP] - pose_coords[RIGHT_HIP])
    torso_height = np.linalg.norm(
        0.5 * (pose_coords[LEFT_SHOULDER] + pose_coords[RIGHT_SHOULDER]) -
        0.5 * (pose_coords[LEFT_HIP] + pose_coords[RIGHT_HIP])
    )

    candidates = [c for c in (shoulder_span, hip_span, torso_height) if c > 1e-4]
    scale = float(np.median(candidates)) if candidates else 1.0

    pose_coords /= scale
    left_hand /= scale
    right_hand /= scale
    face /= scale
    pose[:, :3] = pose_coords

    normalized = np.concatenate([
        pose.reshape(-1),
        left_hand.reshape(-1),
        right_hand.reshape(-1),
        face.reshape(-1)
    ]).astype(np.float32)

    return normalized


def exponential_smooth(sequence: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    if alpha <= 0.0 or sequence.shape[0] < 2:
        return sequence
    smoothed = sequence.copy()
    for idx in range(1, sequence.shape[0]):
        smoothed[idx] = alpha * sequence[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    pose_xyz = frames[:, :POSE_FEATURES].reshape(frames.shape[0], NUM_POSE_LANDMARKS, 4)[..., :3]
    pose_xyz = pose_xyz.reshape(frames.shape[0], -1)
    hands = frames[:, POSE_FEATURES:POSE_FEATURES + 2 * NUM_HAND_LANDMARKS * 3]
    face = frames[:, POSE_FEATURES + 2 * NUM_HAND_LANDMARKS * 3:]
    coords = np.concatenate([pose_xyz, hands, face], axis=1)

    diffs = np.diff(coords, axis=0, prepend=coords[:1])
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


def summarize_motion(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max())
    }


def extract_features_from_video(video_path, face_downsample=FACE_DOWNSAMPLE):
    cap = None
    holistic = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"⚠️ Cannot open video: {video_path}")
            return None, 0.0, 0.0, {}

        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        frames_features = []
        quality_flags = []
        pose_vis_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic.process(frame_rgb)

            frame_feats = []

            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    frame_feats.extend([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.0)])
                pose_visibility = float(np.mean([
                    getattr(lm, 'visibility', 0.0) for lm in result.pose_landmarks.landmark
                ]))
            else:
                frame_feats.extend([0.0] * (NUM_POSE_LANDMARKS * 4))
                pose_visibility = 0.0

            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    frame_feats.extend([lm.x, lm.y, lm.z])
            else:
                frame_feats.extend([0.0] * (NUM_HAND_LANDMARKS * 3))

            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    frame_feats.extend([lm.x, lm.y, lm.z])
            else:
                frame_feats.extend([0.0] * (NUM_HAND_LANDMARKS * 3))

            if result.face_landmarks:
                for i in range(0, 468, face_downsample):
                    lm = result.face_landmarks.landmark[i]
                    frame_feats.extend([lm.x, lm.y, lm.z])
            else:
                frame_feats.extend([0.0] * (NUM_FACE_LANDMARKS * 3))

            if len(frame_feats) != PER_FRAME_FEATURES:
                if len(frame_feats) < PER_FRAME_FEATURES:
                    frame_feats.extend([0.0] * (PER_FRAME_FEATURES - len(frame_feats)))
                else:
                    frame_feats = frame_feats[:PER_FRAME_FEATURES]

            frame_feats = normalize_frame_landmarks(frame_feats)

            has_pose = result.pose_landmarks is not None
            has_hands = (result.left_hand_landmarks is not None) or (result.right_hand_landmarks is not None)

            frames_features.append(frame_feats)
            quality_flags.append(has_pose or has_hands)
            pose_vis_scores.append(pose_visibility)

        total_frames = len(frames_features)
        if total_frames == 0:
            return None, 0.0, 0.0, {}

        frames_array = np.array(frames_features, dtype=np.float32)
        motion_vals = compute_motion_energy(frames_array)
        quality_mask = np.array(quality_flags, dtype=bool)
        visibility_scores = np.array(pose_vis_scores, dtype=np.float32)

        if quality_mask.sum() == 0:
            return None, 0.0, 0.0, {}

        valid_idx = np.where(quality_mask)[0]
        detection_ratio = float(valid_idx.size / total_frames)

        features = frames_array[valid_idx]
        visibility_scores = visibility_scores[valid_idx]
        motion_vals = motion_vals[valid_idx]

        visibility_keep = visibility_scores >= VISIBILITY_THRESHOLD
        motion_keep = motion_vals >= MOTION_KEEP_THRESHOLD
        keep_mask = visibility_keep | motion_keep

        if not keep_mask.any():
            keep_mask[0] = True

        features = features[keep_mask]
        visibility_scores = visibility_scores[keep_mask]
        motion_vals = motion_vals[keep_mask]

        if features.shape[0] == 0:
            return None, detection_ratio, 0.0, {}

        motion_summary = summarize_motion(motion_vals)
        mean_visibility = float(visibility_scores.mean()) if visibility_scores.size else 0.0

        if motion_summary["mean"] < MOTION_REJECTION_THRESHOLD and not KEEP_LOW_QUALITY:
            return None, detection_ratio, mean_visibility, motion_summary

        if features.shape[0] < MIN_FRAMES and not KEEP_LOW_QUALITY:
            return None, detection_ratio, mean_visibility, motion_summary

        features = exponential_smooth(features, alpha=EMA_ALPHA).astype(np.float32)

        return features.astype(np.float32), detection_ratio, mean_visibility, motion_summary

    except Exception as exc:
        print(f"❌ Error processing {video_path}: {exc}")
        traceback.print_exc()
        return None, 0.0, 0.0, {}
    finally:
        if cap is not None:
            cap.release()
        if holistic is not None:
            holistic.close()

print("✅ Feature extraction helpers ready")
```

---

## Cell 5 — Parallel Worker

```python
from multiprocessing import Pool, cpu_count
from functools import partial

def process_single_video(row_dict, video_dir, output_dir, face_downsample):
    import os

    video_file = row_dict['video_file']
    video_path = os.path.join(video_dir, video_file)

    if not os.path.exists(video_path):
        return {
            'status': 'missing',
            'video_file': video_file,
            'error': 'File not found'
        }

    try:
        features, detection_ratio, mean_visibility, motion_summary = extract_features_from_video(
            video_path,
            face_downsample=face_downsample
        )
    except Exception as exc:
        return {
            'status': 'failed',
            'video_file': video_file,
            'error': str(exc)
        }

    if features is None or features.size == 0:
        return {
            'status': 'failed',
            'video_file': video_file,
            'error': 'No usable frames extracted',
            'detection_ratio': float(detection_ratio),
            'mean_visibility': float(mean_visibility),
            'motion_summary': motion_summary or {}
        }

    base_name = os.path.splitext(video_file)[0]
    npy_path = os.path.join(output_dir, f"{base_name}.npy")

    quality_issue = (
        features.shape[0] < MIN_FRAMES
        or detection_ratio < MIN_VALID_RATIO
        or (motion_summary and motion_summary.get('mean', 0.0) < MOTION_REJECTION_THRESHOLD)
    )

    payload = {
        'video_file': video_file,
        'num_frames': int(features.shape[0]),
        'detection_ratio': float(detection_ratio),
        'mean_visibility': float(mean_visibility),
        'motion_summary': motion_summary or {},
        'motion_mean': float(motion_summary.get('mean', 0.0) if motion_summary else 0.0),
        'motion_p50': float(motion_summary.get('p50', 0.0) if motion_summary else 0.0),
        'motion_p95': float(motion_summary.get('p95', 0.0) if motion_summary else 0.0),
        'motion_max': float(motion_summary.get('max', 0.0) if motion_summary else 0.0),
        'sentence_id': row_dict['sentence_id'],
        'sentence': row_dict['sentence'],
        'category': row_dict['category']
    }

    if quality_issue and not KEEP_LOW_QUALITY:
        return {
            **payload,
            'status': 'low_quality',
            'error': 'Below preprocessing quality threshold'
        }

    np.save(npy_path, features.astype(np.float32))
    status = 'success' if not quality_issue else 'low_quality_kept'

    return {
        **payload,
        'status': status,
        'feature_path': npy_path,
        'frame_feature_dim': int(features.shape[1])
    }

print("✅ Parallel processing function defined")
```

---

## Cell 6 — Parallel Execution

```python
num_workers = max(1, cpu_count() - 2)
print("🚀 Starting parallel feature extraction")
print(f"💻 CPU cores: {cpu_count()}")
print(f"👷 Worker processes: {num_workers}")
print(f"📹 Total videos: {len(df_meta)}")
print(f"💾 Output directory: {OUTPUT_DIR}")
print("=" * 60)

process_func = partial(
    process_single_video,
    video_dir=VIDEO_DIR,
    output_dir=OUTPUT_DIR,
    face_downsample=FACE_DOWNSAMPLE
)

results = []
with Pool(num_workers) as pool:
    for result in tqdm(
        pool.imap(process_func, df_meta.to_dict('records')),
        total=len(df_meta),
        desc="Processing videos"
    ):
        results.append(result)

processed_records = []
failed_videos = []
missing_videos = []
low_quality_videos = []

for result in results:
    status = result['status']
    if status == 'success':
        processed_records.append({
            'video_file': result['video_file'],
            'feature_path': result['feature_path'],
            'num_frames': result['num_frames'],
            'frame_feature_dim': result['frame_feature_dim'],
            'detection_ratio': result['detection_ratio'],
            'mean_visibility': result['mean_visibility'],
            'motion_mean': result['motion_mean'],
            'motion_p50': result['motion_p50'],
            'motion_p95': result['motion_p95'],
            'motion_max': result['motion_max'],
            'sentence_id': result['sentence_id'],
            'sentence': result['sentence'],
            'category': result['category']
        })
    elif status == 'low_quality_kept':
        processed_records.append({
            'video_file': result['video_file'],
            'feature_path': result['feature_path'],
            'num_frames': result['num_frames'],
            'frame_feature_dim': result['frame_feature_dim'],
            'detection_ratio': result['detection_ratio'],
            'mean_visibility': result['mean_visibility'],
            'motion_mean': result['motion_mean'],
            'motion_p50': result['motion_p50'],
            'motion_p95': result['motion_p95'],
            'motion_max': result['motion_max'],
            'sentence_id': result['sentence_id'],
            'sentence': result['sentence'],
            'category': result['category'],
            'quality_flag': 'kept_low'
        })
        low_quality_videos.append({
            'video_file': result['video_file'],
            'num_frames': result['num_frames'],
            'detection_ratio': result['detection_ratio'],
            'mean_visibility': result['mean_visibility'],
            'motion_summary': result['motion_summary'],
            'kept': True,
            'sentence_id': result['sentence_id'],
            'sentence': result['sentence'],
            'category': result['category']
        })
    elif status == 'low_quality':
        low_quality_videos.append({
            'video_file': result['video_file'],
            'num_frames': result.get('num_frames'),
            'detection_ratio': result.get('detection_ratio'),
            'mean_visibility': result.get('mean_visibility'),
            'motion_summary': result.get('motion_summary', {}),
            'kept': False,
            'sentence_id': result['sentence_id'],
            'sentence': result['sentence'],
            'category': result['category'],
            'error': result.get('error', 'Below preprocessing quality threshold')
        })
    elif status == 'failed':
        failed_videos.append({
            'video_file': result['video_file'],
            'error': result.get('error', 'Unknown error')
        })
    elif status == 'missing':
        missing_videos.append({
            'video_file': result['video_file'],
            'error': result.get('error', 'File not found')
        })

print("\n" + "=" * 60)
print(f"✅ Processed: {len(processed_records)}")
print(f"❌ Failed: {len(failed_videos)}")
print(f"⚠️ Missing: {len(missing_videos)}")
print(f"⚠️ Low-quality filtered: {sum(not rec.get('kept', False) for rec in low_quality_videos)}")
if KEEP_LOW_QUALITY:
    print(f"   ↳ Low-quality kept (saved): {sum(rec.get('kept', False) for rec in low_quality_videos)}")
```

---

## Cell 7 — Persist Metadata & Diagnostics

```python
df_processed = pd.DataFrame(processed_records)
output_meta_path = "proc/processed_metadata.csv"
df_processed.to_csv(output_meta_path, index=False)

print(f"✅ Saved processed metadata: {output_meta_path}")
print(f"📊 Total records: {len(df_processed)}")

if failed_videos:
    pd.DataFrame(failed_videos).to_csv("proc/failed_videos.csv", index=False)
    print(f"⚠️ Saved failed_videos.csv ({len(failed_videos)} videos)")

if missing_videos:
    pd.DataFrame(missing_videos).to_csv("missing_videos.csv", index=False)
    print(f"⚠️ Saved missing_videos.csv ({len(missing_videos)} videos)")

if low_quality_videos:
    pd.DataFrame(low_quality_videos).to_csv("proc/low_quality_videos.csv", index=False)
    print(f"⚠️ Saved low_quality_videos.csv ({len(low_quality_videos)} clips)")

print("\n📝 Processed metadata preview:")
display(df_processed.head())
```

---

## Cell 8 — Text Augmentation (Gemini API)

This step generates synonymous sentences for each label to support semantic-aware training. It uses the Gemini API to create 3-5 variations per sentence.

```python
# --- TEXT AUGMENTATION PIPELINE ---
import google.generativeai as genai
import time
import random
import json
from collections import defaultdict

# ... (See notebook for full implementation) ...
# Generates 'text_variations' column in processed_metadata.csv
```

---

## Cell 9 — Summary Stats

```python
print("=" * 60)
print("📊 PREPROCESSING SUMMARY")
print("=" * 60)

print(f"\n✅ Successfully processed: {len(df_processed)} videos")
print(f"📁 Feature files saved to: {OUTPUT_DIR}")
print(f"📄 Metadata saved to: {output_meta_path}")

print(f"\n📐 Per-frame dimension: {PER_FRAME_FEATURES}")
print(f"  Average frames/video: {df_processed['num_frames'].mean():.1f}")
print(f"  Median motion mean: {df_processed['motion_mean'].median():.4f}")
print(f"  Median detection ratio: {df_processed['detection_ratio'].median():.2f}")
print(f"  Mean visibility: {df_processed['mean_visibility'].mean():.2f}")

unique_sentences = df_processed['sentence_id'].nunique() if len(df_processed) else 0
print(f"\n📝 Sentence coverage: {unique_sentences}")
if unique_sentences:
    print(f"  Videos per sentence (avg): {len(df_processed) / unique_sentences:.2f}")

removed_low = [rec for rec in low_quality_videos if not rec.get('kept', False)]
if removed_low:
    print(f"\n⚠️ Removed low-motion clips: {len(removed_low)} (see proc/low_quality_videos.csv)")

print("\n✅ Ready for training!")
```

---

## Cell 9 — Global Feature Statistics Export

```python
if len(df_processed) == 0:
    print("⚠️ No processed videos available for statistics")
else:
    total_frames = 0
    sum_feats = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
    sum_sq_feats = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
    seq_lengths = []
    motion_means = []

    for row in tqdm(df_processed.itertuples(), total=len(df_processed), desc="Computing global stats"):
        arr = np.load(row.feature_path).astype(np.float32)
        total_frames += arr.shape[0]
        sum_feats += arr.sum(axis=0)
        sum_sq_feats += (arr ** 2).sum(axis=0)
        seq_lengths.append(arr.shape[0])
        motion_means.append(row.motion_mean)

    if total_frames == 0:
        print("⚠️ Unable to compute stats: total frames = 0")
    else:
        feature_mean = sum_feats / total_frames
        feature_var = sum_sq_feats / total_frames - feature_mean ** 2
        feature_std = np.sqrt(np.maximum(feature_var, 1e-8))

        np.savez(
            GLOBAL_STATS_PATH,
            feature_mean=feature_mean.astype(np.float32),
            feature_std=feature_std.astype(np.float32),
            seq_lengths=np.array(seq_lengths, dtype=np.int32),
            motion_means=np.array(motion_means, dtype=np.float32)
        )

        print("\n✅ Saved global stats:")
        print(f"   Mean/std path: {GLOBAL_STATS_PATH}")
        print(f"   Avg sequence length: {np.mean(seq_lengths):.1f}")
        print(f"   95th percentile length: {np.percentile(seq_lengths, 95):.1f}")
        print(f"   Median motion mean: {np.median(motion_means):.4f}")
```

---

## Optional Visual Checks

Histogram, scatter, and sample feature inspection cells from the previous notebook still apply. They now leverage the richer motion stats in `df_processed` and `proc/low_quality_videos.csv` for quick diagnostics.

---

## Outputs

- `features/pose_data/*.npy` — motion-gated, torso-normalized feature matrices
- `proc/processed_metadata.csv` — metadata plus motion diagnostics for training
- `proc/low_quality_videos.csv` — rejected or low-motion clips
- `proc/global_stats.npz` — dataset mean/std, sequence-length distribution, motion means for training normalization

Once these artefacts exist, move on to the training notebook; it now expects the exported stats and motion metrics.
