```
!pip install mediapipe --no-deps
```

```
!gcloud storage cp -r gs://ghsl-datasets/sample_dataset /content/
```

```
import os
import re
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import traceback
from multiprocessing import Pool

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("✅ Imports successful")
```

```
DATASET_DIR = "/content/sample_dataset"
VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
META_PATH = os.path.join(DATASET_DIR, "sampled_metadata.csv")

# OUTPUT
OUTPUT_DIR = "/content/features/pose_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
GLOBAL_STATS_PATH = "/content/proc/global_stats.npz"
os.makedirs(os.path.dirname(GLOBAL_STATS_PATH), exist_ok=True)

# FEATURE DIMENSIONS
FACE_DOWNSAMPLE = 5  # Sample every 5th face landmark
NUM_FACE_LANDMARKS = len(range(0, 468, FACE_DOWNSAMPLE))  # = 94
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21

# Calculate per-frame feature length
POSE_FEATURES = NUM_POSE_LANDMARKS * 4          # x, y, z, visibility
HANDS_FEATURES = 2 * NUM_HAND_LANDMARKS * 3     # left + right hands
FACE_FEATURES = NUM_FACE_LANDMARKS * 3          # downsampled face
PER_FRAME_FEATURES = POSE_FEATURES + HANDS_FEATURES + FACE_FEATURES

# QUALITY HEURISTICS & TARGET SHAPES
TARGET_SEQ_LEN = 64
MODEL_COMPLEXITY = 2
MIN_FRAMES = 48
MIN_VALID_RATIO = 0.45
EMA_ALPHA = 0.2  # Exponential smoothing factor for temporal jitter
VISIBILITY_THRESHOLD = 0.55
MOTION_KEEP_THRESHOLD = 1e-3
MOTION_REJECTION_THRESHOLD = 7.5e-4
KEEP_LOW_QUALITY = False  # Flip to True to keep flagged clips for manual review

# Landmark indices for torso-centric normalization
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

print("📐 Feature dimensions:")
print(f"  • Pose: {POSE_FEATURES} (33 landmarks × 4)")
print(f"  • Hands: {HANDS_FEATURES} (2 hands × 21 landmarks × 3)")
print(f"  • Face: {FACE_FEATURES} (94 landmarks × 3)")
print(f"  • TOTAL per frame: {PER_FRAME_FEATURES}")
print("⚙️  Quality thresholds")
print(f"  • min_frames: {MIN_FRAMES}")
print(f"  • min_valid_ratio: {MIN_VALID_RATIO}")
print(f"  • visibility min: {VISIBILITY_THRESHOLD}")
print(f"  • motion keep ≥ {MOTION_KEEP_THRESHOLD:.1e}")
print(f"  • motion reject < {MOTION_REJECTION_THRESHOLD:.1e}")
print(f"🧱  Model complexity: {MODEL_COMPLEXITY}")
print(f"📊  Global stats path: {GLOBAL_STATS_PATH}")
```

```
# WORKAROUND CELL - Manual CSV parsing
import csv

META_PATH = "/content/sample_dataset/sampled_metadata.csv"

# Read CSV manually to avoid pandas parsing issues
records = []
with open(META_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        records.append(row)

# Convert to DataFrame manually
df_meta = pd.DataFrame(records)

# Convert sentence_id to int
df_meta['sentence_id'] = df_meta['sentence_id'].astype(int)

print(f"✅ Loaded metadata: {len(df_meta)} videos")
print(f"📋 Columns: {df_meta.columns.tolist()}")
print(f"📊 Unique sentences: {df_meta['sentence_id'].nunique()}")
print(f"📋 Categories: {df_meta['category'].nunique()}")

print("\n📝 Sample rows:")
print(df_meta.head())

# Check for missing values
missing_counts = df_meta.isnull().sum()
print("\nMissing values per column:")
print(missing_counts)
if missing_counts.any():
    print(f"⚠️ Rows with any missing value: {df_meta[df_meta.isnull().any(axis=1)].shape[0]}")
else:
    print("✅ No missing values found.")

# Check that video files exist
missing_files_mask = df_meta['video_file'].apply(lambda x: not os.path.exists(os.path.join(VIDEO_DIR, x)))
missing_files = missing_files_mask.sum()
print(f"Missing video files in {VIDEO_DIR}: {missing_files}")
if missing_files>0:
    print(df_meta[missing_files_mask].head())
```


```
mp_holistic = mp.solutions.holistic

# Global Holistic instance for worker processes (initialized in worker_init)
HOLISTIC = None
FACE_INDICES = list(range(0, 468, FACE_DOWNSAMPLE))


def worker_init(face_downsample: int = FACE_DOWNSAMPLE):
    """Initializer for Pool worker processes: creates a global MediaPipe Holistic instance
    and precomputes face landmark indices. This avoids recreating the model per video.
    """
    global HOLISTIC, FACE_INDICES
    try:
        HOLISTIC = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        HOLISTIC = None
    FACE_INDICES = list(range(0, 468, max(1, int(face_downsample))))


def normalize_frame_landmarks(frame_feats):
    """Center and scale landmarks around the torso to reduce camera variance.

    Optimized to use local variables and avoid repeated attribute lookups.
    Returns a 1D np.float32 vector length PER_FRAME_FEATURES.
    """
    feats = np.asarray(frame_feats, dtype=np.float32)

    # Pose block (33 x 4)
    pose = feats[:POSE_FEATURES].reshape(NUM_POSE_LANDMARKS, 4)

    lh_start = POSE_FEATURES
    lh_end = lh_start + NUM_HAND_LANDMARKS * 3
    rh_end = lh_end + NUM_HAND_LANDMARKS * 3

    left_hand = feats[lh_start:lh_end].reshape(NUM_HAND_LANDMARKS, 3)
    right_hand = feats[lh_end:rh_end].reshape(NUM_HAND_LANDMARKS, 3)
    face = feats[rh_end:rh_end + NUM_FACE_LANDMARKS * 3].reshape(NUM_FACE_LANDMARKS, 3)

    pose_coords = pose[:, :3].copy()

    torso_pts = pose_coords[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]

    # Validate torso points
    if not np.isfinite(torso_pts).all() or np.linalg.norm(torso_pts) < 1e-6:
        return feats.astype(np.float32)

    torso_center = torso_pts.mean(axis=0)

    # Subtract center in-place on local arrays
    pose_coords -= torso_center
    left_hand -= torso_center
    right_hand -= torso_center
    face -= torso_center

    # Compute scale robustly using median of candidate spans
    shoulder_span = np.linalg.norm(pose_coords[LEFT_SHOULDER] - pose_coords[RIGHT_SHOULDER])
    hip_span = np.linalg.norm(pose_coords[LEFT_HIP] - pose_coords[RIGHT_HIP])
    shoulder_center = 0.5 * (pose_coords[LEFT_SHOULDER] + pose_coords[RIGHT_SHOULDER])
    hip_center = 0.5 * (pose_coords[LEFT_HIP] + pose_coords[RIGHT_HIP])
    torso_height = np.linalg.norm(shoulder_center - hip_center)

    candidates = [c for c in (shoulder_span, hip_span, torso_height) if c > 1e-4]
    scale = float(np.median(candidates)) if candidates else 1.0
    if scale < 1e-6:
        scale = 1.0

    inv_scale = 1.0 / scale
    # Apply scale via multiplication (faster than repeated division)
    pose_coords *= inv_scale
    left_hand *= inv_scale
    right_hand *= inv_scale
    face *= inv_scale

    # Put back into pose structure
    pose[:, :3] = pose_coords

    normalized = np.concatenate([
        pose.reshape(-1),
        left_hand.reshape(-1),
        right_hand.reshape(-1),
        face.reshape(-1),
    ]).astype(np.float32)

    return normalized


def exponential_smooth(sequence: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Exponential moving average implemented using an efficient in-place loop.

    For short sequences this loop is inexpensive; vectorized alternatives require
    external dependencies (scipy) so we keep this simple and avoid allocations.
    """
    seq = sequence.astype(np.float32, copy=False)
    n = seq.shape[0]
    if alpha <= 0.0 or n < 2:
        return seq

    out = seq.copy()
    one_minus = 1.0 - alpha
    # operate per-row to keep memory locality
    for i in range(1, n):
        out[i] = alpha * seq[i] + one_minus * out[i - 1]
    return out


def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    """Compute per-frame motion energy from frame-to-frame deltas.

    Optimizations:
    - Uses numpy vectorized diffs
    - Avoids recomputing slice offsets repeatedly
    """
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    # Pose xyz: first POSE_FEATURES entries contain 33*(x,y,z,vis)
    pose_xyz = frames[:, :POSE_FEATURES].reshape(frames.shape[0], NUM_POSE_LANDMARKS, 4)[..., :3]
    pose_flat = pose_xyz.reshape(frames.shape[0], -1)

    hands = frames[:, POSE_FEATURES:POSE_FEATURES + 2 * NUM_HAND_LANDMARKS * 3]
    face = frames[:, POSE_FEATURES + 2 * NUM_HAND_LANDMARKS * 3:]

    coords = np.concatenate([pose_flat, hands, face], axis=1)

    # frame-to-frame differences (prepend first row to keep same length)
    diffs = np.diff(coords, axis=0, prepend=coords[:1])
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


def trim_or_pad_sequence(sequence: np.ndarray, target_len: int = TARGET_SEQ_LEN) -> np.ndarray:
    """Center-crop or pad a sequence to the target length (unchanged behavior)."""
    frames = sequence.shape[0]
    if frames == target_len:
        return sequence
    if frames > target_len:
        start = (frames - target_len) // 2
        return sequence[start:start + target_len]
    pad_len = target_len - frames
    pad = np.zeros((pad_len, sequence.shape[1]), dtype=sequence.dtype)
    return np.vstack([sequence, pad])


print("✅ Feature extraction helpers ready (optimized)")
```

```
from functools import partial
import atexit

# Ensure worker HOLISTIC is closed on process exit
def _close_holistic():
    global HOLISTIC
    try:
        if HOLISTIC is not None:
            HOLISTIC.close()
    except Exception:
        pass

atexit.register(_close_holistic)


def process_single_video(row_dict, video_dir, output_dir, face_downsample):
    """Process a single video, enforcing quality thresholds.

    Uses a per-process global `HOLISTIC` instance when available (created by
    `worker_init`). Falls back to a local instance for single-process runs.
    """
    import os

    video_file = row_dict['video_file']
    video_path = os.path.join(video_dir, video_file)

    if not os.path.exists(video_path):
        return {
            'status': 'missing',
            'video_file': video_file,
            'error': 'File not found'
        }

    local_holistic = None
    holistic_inst = HOLISTIC
    try:
        # Fallback: create a local holistic if worker initializer wasn't used
        if holistic_inst is None:
            local_holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=MODEL_COMPLEXITY,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            holistic_inst = local_holistic

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'status': 'failed', 'video_file': video_file, 'error': 'Cannot open video'}

        frames_features = []
        quality_flags = []
        pose_vis_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic_inst.process(frame_rgb)

            # Build frame feature vector using preallocated list and local variables
            frame_feats = [0.0] * PER_FRAME_FEATURES
            off = 0

            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    frame_feats[off] = lm.x; frame_feats[off + 1] = lm.y; frame_feats[off + 2] = lm.z; frame_feats[off + 3] = getattr(lm, 'visibility', 0.0)
                    off += 4
                pose_visibility = float(np.mean([getattr(lm, 'visibility', 0.0) for lm in result.pose_landmarks.landmark]))
            else:
                off += NUM_POSE_LANDMARKS * 4
                pose_visibility = 0.0

            # Left hand
            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    frame_feats[off] = lm.x; frame_feats[off + 1] = lm.y; frame_feats[off + 2] = lm.z
                    off += 3
            else:
                off += NUM_HAND_LANDMARKS * 3

            # Right hand
            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    frame_feats[off] = lm.x; frame_feats[off + 1] = lm.y; frame_feats[off + 2] = lm.z
                    off += 3
            else:
                off += NUM_HAND_LANDMARKS * 3

            # Face downsampled
            if result.face_landmarks:
                fl = result.face_landmarks.landmark
                for i in range(0, 468, face_downsample):
                    lm = fl[i]
                    frame_feats[off] = lm.x; frame_feats[off + 1] = lm.y; frame_feats[off + 2] = lm.z
                    off += 3
            else:
                off += NUM_FACE_LANDMARKS * 3

            # Safety: trim/pad
            if off > PER_FRAME_FEATURES:
                frame_feats = frame_feats[:PER_FRAME_FEATURES]

            # Normalize around torso to reduce camera variance
            frame_array = normalize_frame_landmarks(frame_feats)

            has_pose = result.pose_landmarks is not None
            has_hands = (result.left_hand_landmarks is not None) or (result.right_hand_landmarks is not None)

            frames_features.append(frame_array)
            quality_flags.append(has_pose or has_hands)
            pose_vis_scores.append(pose_visibility)

        cap.release()

        total_frames = len(frames_features)
        if total_frames == 0:
            return {'status': 'failed', 'video_file': video_file, 'error': 'No frames read'}

        frames_array = np.stack(frames_features, axis=0)
        motion_vals = compute_motion_energy(frames_array)
        quality_mask = np.array(quality_flags, dtype=bool)
        visibility_scores = np.array(pose_vis_scores, dtype=np.float32)

        if quality_mask.sum() == 0:
            return {'status': 'failed', 'video_file': video_file, 'error': 'No valid pose/hand frames'}

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
            return {'status': 'failed', 'video_file': video_file, 'error': 'Filtered to zero frames'}

        motion_summary = summarize_motion(motion_vals)
        mean_visibility = float(visibility_scores.mean()) if visibility_scores.size else 0.0

        if motion_summary.get("mean", 0.0) < MOTION_REJECTION_THRESHOLD and not KEEP_LOW_QUALITY:
            return {'status': 'low_quality', 'video_file': video_file, 'error': 'Low motion', 'detection_ratio': detection_ratio, 'mean_visibility': mean_visibility, 'motion_summary': motion_summary}

        kept_frames = features.shape[0]
        if kept_frames < MIN_FRAMES and not KEEP_LOW_QUALITY:
            return {'status': 'low_quality', 'video_file': video_file, 'error': 'Too few frames', 'detection_ratio': detection_ratio, 'mean_visibility': mean_visibility, 'motion_summary': motion_summary}

        features = exponential_smooth(features, alpha=EMA_ALPHA).astype(np.float32)

        base_name = os.path.splitext(video_file)[0]
        npy_path = os.path.join(output_dir, f"{base_name}.npy")

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

        if (features.shape[0] < MIN_FRAMES
                or detection_ratio < MIN_VALID_RATIO
                or (motion_summary and motion_summary.get('mean', 0.0) < MOTION_REJECTION_THRESHOLD)) and not KEEP_LOW_QUALITY:
            return {**payload, 'status': 'low_quality', 'error': 'Below preprocessing quality threshold'}

        # Save features
        np.save(npy_path, features.astype(np.float32))
        status = 'success'
        return {**payload, 'status': status, 'feature_path': npy_path, 'frame_feature_dim': int(features.shape[1])}

    except Exception as exc:
        return {'status': 'failed', 'video_file': video_file, 'error': str(exc)}

    finally:
        try:
            if local_holistic is not None:
                local_holistic.close()
        except Exception:
            pass
```


```
# Determine number of workers
cpu_count = os.cpu_count()
num_workers = max(1, cpu_count)
print(f"🚀 Starting parallel feature extraction")
print(f"💻 CPU cores: {cpu_count}")
print(f"👷 Worker processes: {num_workers}")
print(f"📹 Total videos: {len(df_meta)}")
print(f"💾 Output directory: {OUTPUT_DIR}")
print("=" * 60)

# Convert DataFrame to list of dicts for multiprocessing
video_records = df_meta.to_dict('records')

# Create partial function with fixed parameters
process_func = partial(
    process_single_video,
    video_dir=VIDEO_DIR,
    output_dir=OUTPUT_DIR,
    face_downsample=FACE_DOWNSAMPLE,
)

# Process videos in parallel with progress bar, reusing MediaPipe per worker
results = []
with Pool(num_workers, initializer=worker_init, initargs=(FACE_DOWNSAMPLE,)) as pool:
    for result in tqdm(
        pool.imap(process_func, video_records),
        total=len(video_records),
        desc="Processing videos"
    ):
        results.append(result)

# Separate results by status
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
            'motion_mean': result['motion_mean'],
            'motion_p50': result['motion_p50'],
            'motion_p95': result['motion_p95'],
            'motion_max': result['motion_max'],
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
            'motion_mean': result.get('motion_mean'),
            'motion_p50': result.get('motion_p50'),
            'motion_p95': result.get('motion_p95'),
            'motion_max': result.get('motion_max'),
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

```
# Save processed metadata
df_processed = pd.DataFrame(processed_records)
output_meta_path = "proc/processed_metadata.csv"
df_processed.to_csv(output_meta_path, index=False)

print(f"✅ Saved processed metadata: {output_meta_path}")
print(f"📊 Total records: {len(df_processed)}")

# Save debug info for failed/missing videos
if failed_videos:
    pd.DataFrame(failed_videos).to_csv("proc/failed_videos.csv", index=False)
    print(f"⚠️ Saved failed_videos.csv ({len(failed_videos)} videos)")

if missing_videos:
    pd.DataFrame(missing_videos).to_csv("missing_videos.csv", index=False)
    print(f"⚠️ Saved missing_videos.csv ({len(missing_videos)} videos)")

if low_quality_videos:
    pd.DataFrame(low_quality_videos).to_csv("proc/low_quality_videos.csv", index=False)
    print(f"⚠️ Saved low_quality_videos.csv ({len(low_quality_videos)} clips)")

print("\n📝 Processed metadata sample:")
df_processed.head()
```

```
if not df_processed.empty:
    all_features = []
    for feature_path in df_processed['feature_path']:
        feats = np.load(feature_path)
        all_features.append(feats)

    all_features_concat = np.vstack(all_features)
    global_mean = np.mean(all_features_concat, axis=0)
    global_std = np.std(all_features_concat, axis=0)

    np.savez_compressed(
        GLOBAL_STATS_PATH,
        mean=global_mean.astype(np.float32),
        std=global_std.astype(np.float32)
    )

    print(f"✅ Saved global stats: {GLOBAL_STATS_PATH}")
    print(f"  • Feature dimension: {global_mean.shape[0]}")

```


```
# Copy the processed features into the `ghsl-datasets` bucket
!gcloud storage cp -r /content/features gs://ghsl-datasets/
!gcloud storage cp -r /content/proc gs://ghsl-datasets/
```

```
import google.generativeai as genai
import pandas as pd
import os
import time
import random
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
DEFAULT_MODEL = "gemini-2.0-flash"
WORKER_MODEL_POOL = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]
MODEL_NAME = DEFAULT_MODEL  # Backward compat
CACHE_PATH = "proc/sentence_variations.json"
INPUT_CSV = "proc/processed_metadata.csv"
KEYS_PER_WORKER = 2  # 2 keys per worker

def load_env_keys(env_path=".env"):
    """Loads keys from .env file."""
    keys = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    _, v = line.split("=", 1)
                    if v:
                        keys.append(v)
    return keys

# --- WORKER FUNCTION (Runs in a separate process) ---
def process_batch(batch_data, worker_keys, worker_model, worker_id):
    """Handles a chunk of sentences using a specific set of API keys and model."""
    current_key_idx = 0
    model_name = worker_model or DEFAULT_MODEL
    genai.configure(api_key=worker_keys[current_key_idx])

    local_results = {}

    def rotate_key():
        nonlocal current_key_idx
        if len(worker_keys) > 1:
            current_key_idx = (current_key_idx + 1) % len(worker_keys)
            genai.configure(api_key=worker_keys[current_key_idx])
            print(f"[Worker {worker_id}] 🔄 Rotated to Key #{current_key_idx}", flush=True)

    def generate_synonyms_safe(sentence, retries=3):
        prompt = f"""
        Generate 5 distinct, natural-sounding English variations of the following sign language gloss/sentence.
        Keep the meaning identical but vary the phrasing (formal, casual, short, descriptive).
        Output ONLY a JSON array of strings.
        Input: "{sentence}"
        """

        backoff = 2
        for attempt in range(retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)

                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                variations = json.loads(text)
                if isinstance(variations, list):
                    return variations

            except Exception as e:
                err_str = str(e)
                print(f"[Worker {worker_id}] ⚠️ Error: {err_str}", flush=True)
                if "429" in err_str or "ResourceExhausted" in err_str:
                    rotate_key()
                    time.sleep(backoff + random.uniform(0, 1))
                    backoff = min(backoff * 2, 16)
                else:
                    break

        return [sentence]

    processed_count = 0
    print(f"[Worker {worker_id}] Using model: {model_name}", flush=True)
    for sid, sentence in batch_data:
        variations = generate_synonyms_safe(sentence)
        if sentence not in variations:
            variations.insert(0, sentence)
        local_results[sid] = variations
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"[Worker {worker_id}] Progress: {processed_count}/{len(batch_data)}", flush=True)
        time.sleep(1.0)

    print(f"[Worker {worker_id}] Done chunk: {processed_count} items", flush=True)
    return local_results

# --- MAIN CONTROLLER ---
def main():
    print("=" * 60)
    print("🤖 STARTING PARALLEL TEXT AUGMENTATION")
    print("=" * 60)

    # Load keys (prefer env, fallback to hardcoded list if present)
    API_KEYS = [
        "Key_1", "Key_2",
        "Key_3", "Key_4",
        "Key_5", "Key_6"
    ]

    num_total_keys = len(API_KEYS)
    if num_total_keys < KEYS_PER_WORKER:
        print(f"⚠️ Not enough keys for parallel processing. Need at least {KEYS_PER_WORKER}.")
        return

    num_workers = num_total_keys // KEYS_PER_WORKER
    print(f"✅ Found {num_total_keys} API Keys.")
    print(f"🚀 Spawning {num_workers} parallel workers (Process Isolation).")

    df = pd.read_csv(INPUT_CSV)
    unique_sentences_df = df[['sentence_id', 'sentence']].drop_duplicates()

    sentence_map = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            sentence_map = json.load(f)
        print(f"✅ Loaded {len(sentence_map)} cached variations.")

    to_process = []
    for _, row in unique_sentences_df.iterrows():
        sid = str(row['sentence_id'])
        if sid not in sentence_map:
            to_process.append((sid, row['sentence']))

    total_items = len(to_process)
    print(f"🔠 Sentences remaining to process: {total_items}")

    if total_items == 0:
        print("🎉 Nothing to do!")
        return

    chunk_size = math.ceil(total_items / num_workers)
    chunks = [to_process[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

    def model_for_worker(worker_idx):
        if WORKER_MODEL_POOL:
            return WORKER_MODEL_POOL[worker_idx % len(WORKER_MODEL_POOL)]
        return DEFAULT_MODEL

    futures = []
    results_aggregated = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(num_workers):
            key_subset = API_KEYS[i*KEYS_PER_WORKER : (i+1)*KEYS_PER_WORKER]
            if i < len(chunks):
                worker_model = model_for_worker(i)
                print(f"[Main] Submitting chunk {i+1}/{len(chunks)} with {len(chunks[i])} items using model {worker_model}", flush=True)
                future = executor.submit(process_batch, chunks[i], key_subset, worker_model, i+1)
                futures.append(future)

        print("\n⏳ Processing in parallel... (This might take a while)")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Worker Progress"):
            try:
                worker_result = future.result()
                results_aggregated.update(worker_result)
                sentence_map.update(worker_result)
                with open(CACHE_PATH, "w") as f:
                    json.dump(sentence_map, f, indent=2)
                print(f"[Main] ✅ Merged chunk; total cached: {len(sentence_map)}", flush=True)
            except Exception as e:
                print(f"❌ A worker failed: {e}", flush=True)

    print(f"✅ Processing complete. Total processed: {len(sentence_map)}")

    def get_variations(sid):
        sid_str = str(sid)
        if sid_str in sentence_map:
            return json.dumps(sentence_map[sid_str])
        return json.dumps([df.loc[df['sentence_id']==sid, 'sentence'].iloc[0]])

    df['text_variations'] = df['sentence_id'].apply(get_variations)
    df.to_csv(INPUT_CSV, index=False)
    print(f"💾 Saved final metadata to {INPUT_CSV}")

if __name__ == "__main__":
    main()
```


```
print("=" * 60)
print("📊 PREPROCESSING SUMMARY")
print("=" * 60)

print(f"\n✅ Successfully processed: {len(df_processed)} videos")
print(f"📁 Feature files saved to: {OUTPUT_DIR}")
print(f"📄 Metadata saved to: proc/processed_metadata.csv")

print(f"\n📐 Feature dimensions:")
print(f"  • Per-frame features: {PER_FRAME_FEATURES}")
print(f"  • Shape: (num_frames, {PER_FRAME_FEATURES})")

print(f"\n📊 Frame statistics:")
print(f"  • Average frames per video: {df_processed['num_frames'].mean():.1f}")
print(f"  • Min frames: {df_processed['num_frames'].min()}")
print(f"  • Max frames: {df_processed['num_frames'].max()}")
print(f"  • Median frames: {df_processed['num_frames'].median():.1f}")
if 'detection_ratio' in df_processed.columns:
    print(f"  • Median detection ratio: {df_processed['detection_ratio'].median():.2f}")
    print(f"  • Mean pose visibility: {df_processed['mean_visibility'].mean():.2f}")
if 'motion_mean' in df_processed.columns:
    print(f"  • Median motion energy: {df_processed['motion_mean'].median():.4f}")
    print(f"  • 95th percentile motion: {df_processed['motion_p95'].median():.4f}")

print(f"\n📝 Sentence coverage:")
print(f"  • Unique sentences: {df_processed['sentence_id'].nunique()}")
print(f"  • Videos per sentence (avg): {len(df_processed) / df_processed['sentence_id'].nunique():.2f}")

print(f"\n📋 Category distribution:")
print(df_processed['category'].value_counts())

removed_low = [rec for rec in low_quality_videos if not rec.get('kept', False)]
if removed_low:
    print(f"\n⚠️ Removed low-motion/low-visibility clips: {len(removed_low)} (see proc/low_quality_videos.csv)")

print("\n✅ Ready for model training!")
```


```
# Compute global feature statistics for consistent normalization
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
        if hasattr(row, 'motion_mean'):
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
        print(f"   • Mean/std saved to {GLOBAL_STATS_PATH}")
        print(f"   • Avg sequence length: {np.mean(seq_lengths):.1f}")
        print(f"   • 95th percentile length: {np.percentile(seq_lengths, 95):.1f}")
        if motion_means:
            print(f"   • Median motion mean: {np.median(motion_means):.4f}")
```


```
#Copy all relevant files that were modified,
!gcloud storage cp proc/processed_metadata.csv gs://ghsl-datasets/proc/processed_metadata.csv
!gcloud storage cp proc/low_quality_videos.csv gs://ghsl-datasets/proc/low_quality_videos.csv
!gcloud storage cp proc/global_stats.npz gs://ghsl-datasets/proc/global_stats.npz
```

```
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


```
# Test loading a random .npy file
sample_row = df_processed.sample(1).iloc[0]

print(f"📹 Video: {sample_row['video_file']}")
print(f"📝 Sentence: {sample_row['sentence']}")
print(f"📋 Category: {sample_row['category']}")
print(f"🎬 Frames: {sample_row['num_frames']}")
if 'detection_ratio' in sample_row:
    print(f"📈 Detection ratio: {sample_row['detection_ratio']:.2f}")
    print(f"👁️ Mean pose visibility: {sample_row['mean_visibility']:.2f}")

features = np.load(sample_row['feature_path'])
print(f"\n✅ Loaded features shape: {features.shape}")
print(f"   Expected: ({sample_row['num_frames']}, {PER_FRAME_FEATURES})")

print(f"\n📊 First frame sample (first 10 values):")
print(features[0, :10])
print(f"   Min: {features.min():.4f}, Max: {features.max():.4f}")
```