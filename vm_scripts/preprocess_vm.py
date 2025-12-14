"""
End-to-end preprocessing script adapted from the GCP Colab notebook.
- Downloads dataset from GCS
- Extracts MediaPipe pose/hand/face features in parallel
- Saves per-video .npy features and processed metadata
- Computes global stats and optional text variations augmentation
- Optionally uploads outputs back to GCS
- Exports plots instead of showing them interactively
"""
import argparse
import atexit
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple

import cv2
import google.generativeai as genai
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress TensorFlow warnings used internally by MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------
# Configuration defaults
# -----------------------------
DEFAULT_DATASET_GCS_URI = "gs://ghsl-datasets/sample_dataset"
DEFAULT_WORKDIR = "/mnt/disks/data"
DEFAULT_OUTPUT_GCS_URI = "gs://ghsl-model-artifacts/sign2text"
DEFAULT_FACE_DOWNSAMPLE = 5
TARGET_SEQ_LEN = 64
MODEL_COMPLEXITY = 2
MIN_FRAMES = 48
MIN_VALID_RATIO = 0.45
EMA_ALPHA = 0.2
VISIBILITY_THRESHOLD = 0.55
MOTION_KEEP_THRESHOLD = 1e-3
MOTION_REJECTION_THRESHOLD = 7.5e-4
KEEP_LOW_QUALITY_DEFAULT = False
KEYS_PER_WORKER = 2
DEFAULT_MODEL = "gemini-2.0-flash"
WORKER_MODEL_POOL = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

# MediaPipe globals for worker processes
mp_holistic = mp.solutions.holistic
HOLISTIC = None
FACE_INDICES: List[int] = []


# -----------------------------
# Utility helpers
# -----------------------------
def run_cmd(cmd: List[str]) -> None:
    """Run a shell command and raise on failure."""
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("Command failed: %s", result.stderr.strip())
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    if result.stdout:
        logging.debug(result.stdout.strip())


def download_dataset(gcs_uri: str, dest_dir: str) -> str:
    """Download dataset folder from GCS into dest_dir and return local dataset path."""
    os.makedirs(dest_dir, exist_ok=True)
    run_cmd(["gcloud", "storage", "cp", "-r", gcs_uri, dest_dir])
    folder_name = os.path.basename(gcs_uri.rstrip("/"))
    dataset_path = os.path.join(dest_dir, folder_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Expected dataset at {dataset_path} after download")
    return dataset_path


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_env_keys(env_path: str) -> List[str]:
    """Loads API keys from .env-style file (KEY=value)."""
    keys: List[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    _, v = line.split("=", 1)
                    if v:
                        keys.append(v)
    return keys


# -----------------------------
# Feature extraction helpers
# -----------------------------
FACE_DOWNSAMPLE = DEFAULT_FACE_DOWNSAMPLE
NUM_FACE_LANDMARKS = len(range(0, 468, DEFAULT_FACE_DOWNSAMPLE))
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
POSE_FEATURES = NUM_POSE_LANDMARKS * 4
HANDS_FEATURES = 2 * NUM_HAND_LANDMARKS * 3
FACE_FEATURES = NUM_FACE_LANDMARKS * 3
PER_FRAME_FEATURES = POSE_FEATURES + HANDS_FEATURES + FACE_FEATURES
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def worker_init(face_downsample: int = DEFAULT_FACE_DOWNSAMPLE):
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


def normalize_frame_landmarks(frame_feats: List[float]) -> np.ndarray:
    feats = np.asarray(frame_feats, dtype=np.float32)
    pose = feats[:POSE_FEATURES].reshape(NUM_POSE_LANDMARKS, 4)
    lh_start = POSE_FEATURES
    lh_end = lh_start + NUM_HAND_LANDMARKS * 3
    rh_end = lh_end + NUM_HAND_LANDMARKS * 3
    left_hand = feats[lh_start:lh_end].reshape(NUM_HAND_LANDMARKS, 3)
    right_hand = feats[lh_end:rh_end].reshape(NUM_HAND_LANDMARKS, 3)
    face = feats[rh_end:rh_end + NUM_FACE_LANDMARKS * 3].reshape(NUM_FACE_LANDMARKS, 3)
    pose_coords = pose[:, :3].copy()
    torso_pts = pose_coords[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]
    if not np.isfinite(torso_pts).all() or np.linalg.norm(torso_pts) < 1e-6:
        return feats.astype(np.float32)
    torso_center = torso_pts.mean(axis=0)
    pose_coords -= torso_center
    left_hand -= torso_center
    right_hand -= torso_center
    face -= torso_center
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
    pose_coords *= inv_scale
    left_hand *= inv_scale
    right_hand *= inv_scale
    face *= inv_scale
    pose[:, :3] = pose_coords
    normalized = np.concatenate([pose.reshape(-1), left_hand.reshape(-1), right_hand.reshape(-1), face.reshape(-1)])
    return normalized.astype(np.float32)


def exponential_smooth(sequence: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    seq = sequence.astype(np.float32, copy=False)
    n = seq.shape[0]
    if alpha <= 0.0 or n < 2:
        return seq
    out = seq.copy()
    one_minus = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * seq[i] + one_minus * out[i - 1]
    return out


def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    pose_xyz = frames[:, :POSE_FEATURES].reshape(frames.shape[0], NUM_POSE_LANDMARKS, 4)[..., :3]
    pose_flat = pose_xyz.reshape(frames.shape[0], -1)
    hands = frames[:, POSE_FEATURES:POSE_FEATURES + 2 * NUM_HAND_LANDMARKS * 3]
    face = frames[:, POSE_FEATURES + 2 * NUM_HAND_LANDMARKS * 3:]
    coords = np.concatenate([pose_flat, hands, face], axis=1)
    diffs = np.diff(coords, axis=0, prepend=coords[:1])
    energy = np.linalg.norm(diffs, axis=1)
    return energy.astype(np.float32)


def summarize_motion(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


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


# Ensure worker HOLISTIC is closed on exit
@atexit.register
def _close_holistic():
    global HOLISTIC
    try:
        if HOLISTIC is not None:
            HOLISTIC.close()
    except Exception:
        pass


# -----------------------------
# Core processing
# -----------------------------
def process_single_video(row_dict: Dict, video_dir: str, output_dir: str, face_downsample: int, keep_low_quality: bool):
    import os as _os

    video_file = row_dict["video_file"]
    video_path = _os.path.join(video_dir, video_file)
    if not _os.path.exists(video_path):
        return {"status": "missing", "video_file": video_file, "error": "File not found"}

    local_holistic = None
    holistic_inst = HOLISTIC
    try:
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
            return {"status": "failed", "video_file": video_file, "error": "Cannot open video"}

        frames_features: List[np.ndarray] = []
        quality_flags: List[bool] = []
        pose_vis_scores: List[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = holistic_inst.process(frame_rgb)
            frame_feats = [0.0] * PER_FRAME_FEATURES
            off = 0
            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    frame_feats[off] = lm.x
                    frame_feats[off + 1] = lm.y
                    frame_feats[off + 2] = lm.z
                    frame_feats[off + 3] = getattr(lm, "visibility", 0.0)
                    off += 4
                pose_visibility = float(np.mean([getattr(lm, "visibility", 0.0) for lm in result.pose_landmarks.landmark]))
            else:
                off += NUM_POSE_LANDMARKS * 4
                pose_visibility = 0.0
            if result.left_hand_landmarks:
                for lm in result.left_hand_landmarks.landmark:
                    frame_feats[off] = lm.x
                    frame_feats[off + 1] = lm.y
                    frame_feats[off + 2] = lm.z
                    off += 3
            else:
                off += NUM_HAND_LANDMARKS * 3
            if result.right_hand_landmarks:
                for lm in result.right_hand_landmarks.landmark:
                    frame_feats[off] = lm.x
                    frame_feats[off + 1] = lm.y
                    frame_feats[off + 2] = lm.z
                    off += 3
            else:
                off += NUM_HAND_LANDMARKS * 3
            if result.face_landmarks:
                fl = result.face_landmarks.landmark
                for i in range(0, 468, face_downsample):
                    lm = fl[i]
                    frame_feats[off] = lm.x
                    frame_feats[off + 1] = lm.y
                    frame_feats[off + 2] = lm.z
                    off += 3
            else:
                off += NUM_FACE_LANDMARKS * 3
            if off > PER_FRAME_FEATURES:
                frame_feats = frame_feats[:PER_FRAME_FEATURES]
            frame_array = normalize_frame_landmarks(frame_feats)
            has_pose = result.pose_landmarks is not None
            has_hands = (result.left_hand_landmarks is not None) or (result.right_hand_landmarks is not None)
            frames_features.append(frame_array)
            quality_flags.append(has_pose or has_hands)
            pose_vis_scores.append(pose_visibility)

        cap.release()
        total_frames = len(frames_features)
        if total_frames == 0:
            return {"status": "failed", "video_file": video_file, "error": "No frames read"}

        frames_array = np.stack(frames_features, axis=0)
        motion_vals = compute_motion_energy(frames_array)
        quality_mask = np.array(quality_flags, dtype=bool)
        visibility_scores = np.array(pose_vis_scores, dtype=np.float32)

        if quality_mask.sum() == 0:
            return {"status": "failed", "video_file": video_file, "error": "No valid pose/hand frames"}

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
            return {"status": "failed", "video_file": video_file, "error": "Filtered to zero frames"}

        motion_summary = summarize_motion(motion_vals)
        mean_visibility = float(visibility_scores.mean()) if visibility_scores.size else 0.0
        if motion_summary.get("mean", 0.0) < MOTION_REJECTION_THRESHOLD and not keep_low_quality:
            return {
                "status": "low_quality",
                "video_file": video_file,
                "error": "Low motion",
                "detection_ratio": detection_ratio,
                "mean_visibility": mean_visibility,
                "motion_summary": motion_summary,
                "sentence_id": row_dict["sentence_id"],
                "sentence": row_dict["sentence"],
                "category": row_dict["category"],
            }
        kept_frames = features.shape[0]
        if kept_frames < MIN_FRAMES and not keep_low_quality:
            return {
                "status": "low_quality",
                "video_file": video_file,
                "error": "Too few frames",
                "detection_ratio": detection_ratio,
                "mean_visibility": mean_visibility,
                "motion_summary": motion_summary,
                "sentence_id": row_dict["sentence_id"],
                "sentence": row_dict["sentence"],
                "category": row_dict["category"],
            }

        features = exponential_smooth(features, alpha=EMA_ALPHA).astype(np.float32)
        base_name = _os.path.splitext(video_file)[0]
        npy_path = _os.path.join(output_dir, f"{base_name}.npy")
        payload = {
            "video_file": video_file,
            "num_frames": int(features.shape[0]),
            "detection_ratio": float(detection_ratio),
            "mean_visibility": float(mean_visibility),
            "motion_summary": motion_summary or {},
            "motion_mean": float(motion_summary.get("mean", 0.0) if motion_summary else 0.0),
            "motion_p50": float(motion_summary.get("p50", 0.0) if motion_summary else 0.0),
            "motion_p95": float(motion_summary.get("p95", 0.0) if motion_summary else 0.0),
            "motion_max": float(motion_summary.get("max", 0.0) if motion_summary else 0.0),
            "sentence_id": row_dict["sentence_id"],
            "sentence": row_dict["sentence"],
            "category": row_dict["category"],
        }
        if (features.shape[0] < MIN_FRAMES or detection_ratio < MIN_VALID_RATIO or (motion_summary and motion_summary.get("mean", 0.0) < MOTION_REJECTION_THRESHOLD)) and not keep_low_quality:
            return {**payload, "status": "low_quality", "error": "Below preprocessing quality threshold"}
        np.save(npy_path, features.astype(np.float32))
        status = "success"
        return {**payload, "status": status, "feature_path": npy_path, "frame_feature_dim": int(features.shape[1])}

    except Exception as exc:
        return {"status": "failed", "video_file": video_file, "error": str(exc)}

    finally:
        try:
            if local_holistic is not None:
                local_holistic.close()
        except Exception:
            pass


# -----------------------------
# Text augmentation (optional)
# -----------------------------
def process_batch(batch_data: List[Tuple[str, str]], worker_keys: List[str], worker_model: str, worker_id: int) -> Dict[str, List[str]]:
    current_key_idx = 0
    model_name = worker_model or DEFAULT_MODEL
    genai.configure(api_key=worker_keys[current_key_idx])
    local_results: Dict[str, List[str]] = {}

    def rotate_key():
        nonlocal current_key_idx
        if len(worker_keys) > 1:
            current_key_idx = (current_key_idx + 1) % len(worker_keys)
            genai.configure(api_key=worker_keys[current_key_idx])
            logging.info("[Worker %s] Rotated to Key #%s", worker_id, current_key_idx)

    def generate_synonyms_safe(sentence: str, retries: int = 3) -> List[str]:
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
                logging.warning("[Worker %s] Error: %s", worker_id, err_str)
                if "429" in err_str or "ResourceExhausted" in err_str:
                    rotate_key()
                    time.sleep(backoff + random.uniform(0, 1))
                    backoff = min(backoff * 2, 16)
                else:
                    break
        return [sentence]

    processed_count = 0
    logging.info("[Worker %s] Using model: %s", worker_id, model_name)
    for sid, sentence in batch_data:
        variations = generate_synonyms_safe(sentence)
        if sentence not in variations:
            variations.insert(0, sentence)
        local_results[sid] = variations
        processed_count += 1
        if processed_count % 10 == 0:
            logging.info("[Worker %s] Progress: %s/%s", worker_id, processed_count, len(batch_data))
        time.sleep(1.0)
    logging.info("[Worker %s] Done chunk: %s items", worker_id, processed_count)
    return local_results


# -----------------------------
# Main pipeline
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose/hand/face preprocessing pipeline for GCE VM")
    parser.add_argument("--dataset-gcs-uri", default=DEFAULT_DATASET_GCS_URI, help="GCS URI for input dataset folder")
    parser.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Local working directory on the VM")
    parser.add_argument("--output-gcs-uri", default=DEFAULT_OUTPUT_GCS_URI, help="GCS URI to upload outputs (features and proc)")
    parser.add_argument("--face-downsample", type=int, default=DEFAULT_FACE_DOWNSAMPLE, help="Face landmark downsample factor")
    parser.add_argument("--keep-low-quality", action="store_true", default=KEEP_LOW_QUALITY_DEFAULT, help="Keep low-quality clips instead of filtering")
    parser.add_argument("--max-workers", type=int, default=None, help="Override worker count (defaults to CPU cores)")
    parser.add_argument("--augment-text", action="store_true", help="Run text variation augmentation with Gemini")
    parser.add_argument("--env-path", default=".env", help="Path to .env file containing Gemini keys")
    parser.add_argument("--plot-dir", default=None, help="Directory to save plots (defaults to <workdir>/proc/plots)")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(message)s")
    gcs_base = args.output_gcs_uri.rstrip("/") if args.output_gcs_uri else None
    global FACE_DOWNSAMPLE, NUM_FACE_LANDMARKS, FACE_FEATURES
    FACE_DOWNSAMPLE = args.face_downsample
    NUM_FACE_LANDMARKS = len(range(0, 468, FACE_DOWNSAMPLE))
    FACE_FEATURES = NUM_FACE_LANDMARKS * 3

    logging.info("Starting preprocessing pipeline")
    dataset_dir = download_dataset(args.dataset_gcs_uri, args.workdir)
    video_dir = os.path.join(dataset_dir, "videos")
    meta_path = os.path.join(dataset_dir, "sampled_metadata.csv")
    output_dir = os.path.join(args.workdir, "features", "pose_data")
    proc_dir = os.path.join(args.workdir, "proc")
    plot_dir = args.plot_dir or os.path.join(proc_dir, "plots")
    ensure_dirs(output_dir, proc_dir, plot_dir)
    global_stats_path = os.path.join(proc_dir, "global_stats.npz")

    logging.info("Loading metadata from %s", meta_path)
    records: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    df_meta = pd.DataFrame(records)
    df_meta["sentence_id"] = df_meta["sentence_id"].astype(int)
    logging.info("Loaded %s videos", len(df_meta))
    missing_files_mask = df_meta["video_file"].apply(lambda x: not os.path.exists(os.path.join(video_dir, x)))
    missing_files = missing_files_mask.sum()
    if missing_files:
        logging.warning("Missing video files: %s", missing_files)
    cpu_count = os.cpu_count() or 1
    num_workers = args.max_workers or cpu_count
    logging.info("Workers: %s", num_workers)

    video_records = df_meta.to_dict("records")
    process_func = partial(
        process_single_video,
        video_dir=video_dir,
        output_dir=output_dir,
        face_downsample=FACE_DOWNSAMPLE,
        keep_low_quality=args.keep_low_quality,
    )
    results: List[Dict] = []
    with Pool(num_workers, initializer=worker_init, initargs=(FACE_DOWNSAMPLE,)) as pool:
        for result in tqdm(pool.imap(process_func, video_records), total=len(video_records), desc="Processing videos"):
            results.append(result)

    processed_records: List[Dict] = []
    failed_videos: List[Dict] = []
    missing_videos: List[Dict] = []
    low_quality_videos: List[Dict] = []
    for result in results:
        status = result["status"]
        if status == "success":
            processed_records.append({
                "video_file": result["video_file"],
                "feature_path": result["feature_path"],
                "num_frames": result["num_frames"],
                "frame_feature_dim": result["frame_feature_dim"],
                "detection_ratio": result["detection_ratio"],
                "mean_visibility": result["mean_visibility"],
                "motion_mean": result["motion_mean"],
                "motion_p50": result["motion_p50"],
                "motion_p95": result["motion_p95"],
                "motion_max": result["motion_max"],
                "sentence_id": result["sentence_id"],
                "sentence": result["sentence"],
                "category": result["category"],
            })
        elif status == "low_quality":
            low_quality_videos.append({
                "video_file": result["video_file"],
                "num_frames": result.get("num_frames"),
                "detection_ratio": result.get("detection_ratio"),
                "mean_visibility": result.get("mean_visibility"),
                "motion_mean": result.get("motion_mean"),
                "motion_p50": result.get("motion_p50"),
                "motion_p95": result.get("motion_p95"),
                "motion_max": result.get("motion_max"),
                "motion_summary": result.get("motion_summary", {}),
                "kept": False,
                "sentence_id": result.get("sentence_id"),
                "sentence": result.get("sentence"),
                "category": result.get("category"),
                "error": result.get("error", "Below preprocessing quality threshold"),
            })
        elif status == "failed":
            failed_videos.append({"video_file": result["video_file"], "error": result.get("error", "Unknown error")})
        elif status == "missing":
            missing_videos.append({"video_file": result["video_file"], "error": result.get("error", "File not found")})

    logging.info("Processed: %s", len(processed_records))
    logging.info("Failed: %s", len(failed_videos))
    logging.info("Missing: %s", len(missing_videos))
    logging.info("Low-quality filtered: %s", len(low_quality_videos))

    df_processed = pd.DataFrame(processed_records)
    if gcs_base and not df_processed.empty:
        gcs_features = f"{gcs_base}/features"
        df_processed["feature_path_local"] = df_processed["feature_path"]
        df_processed["feature_path"] = df_processed["feature_path"].apply(
            lambda p: f"{gcs_features}/pose_data/{os.path.basename(p)}"
        )
        df_processed["feature_path_gcs"] = df_processed["feature_path"]

    output_meta_path = os.path.join(proc_dir, "processed_metadata.csv")
    df_processed.to_csv(output_meta_path, index=False)
    logging.info("Saved processed metadata to %s", output_meta_path)

    if failed_videos:
        pd.DataFrame(failed_videos).to_csv(os.path.join(proc_dir, "failed_videos.csv"), index=False)
    if missing_videos:
        pd.DataFrame(missing_videos).to_csv(os.path.join(proc_dir, "missing_videos.csv"), index=False)
    if low_quality_videos:
        pd.DataFrame(low_quality_videos).to_csv(os.path.join(proc_dir, "low_quality_videos.csv"), index=False)

    if len(df_processed) > 0:
        total_frames = 0
        sum_feats = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
        sum_sq_feats = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
        seq_lengths: List[int] = []
        motion_means: List[float] = []
        for row in tqdm(df_processed.itertuples(), total=len(df_processed), desc="Computing global stats"):
            arr = np.load(row.feature_path).astype(np.float32)
            total_frames += arr.shape[0]
            sum_feats += arr.sum(axis=0)
            sum_sq_feats += (arr ** 2).sum(axis=0)
            seq_lengths.append(arr.shape[0])
            if hasattr(row, "motion_mean"):
                motion_means.append(row.motion_mean)
        if total_frames > 0:
            feature_mean = sum_feats / total_frames
            feature_var = sum_sq_feats / total_frames - feature_mean ** 2
            feature_std = np.sqrt(np.maximum(feature_var, 1e-8))
            np.savez(
                global_stats_path,
                feature_mean=feature_mean.astype(np.float32),
                feature_std=feature_std.astype(np.float32),
                seq_lengths=np.array(seq_lengths, dtype=np.int32),
                motion_means=np.array(motion_means, dtype=np.float32),
            )
            logging.info("Saved global stats to %s", global_stats_path)
        else:
            logging.warning("No frames available to compute global stats")

    # Export plots instead of interactive display
    if len(df_processed) > 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        df_processed["num_frames"].hist(bins=30, edgecolor="black")
        plt.xlabel("Number of Frames")
        plt.ylabel("Frequency")
        plt.title("Distribution of Frame Counts")
        plt.axvline(df_processed["num_frames"].mean(), color="red", linestyle="--", label="Mean")
        plt.legend()
        plt.subplot(1, 2, 2)
        category_counts = df_processed["category"].value_counts()
        category_counts.plot(kind="barh")
        plt.xlabel("Number of Videos")
        plt.title("Videos per Category")
        plt.tight_layout()
        frames_plot_path = os.path.join(plot_dir, "frames_distribution.png")
        plt.savefig(frames_plot_path, dpi=150)
        logging.info("Saved plot: %s", frames_plot_path)
        plt.close("all")

    # Optional text augmentation
    if args.augment_text:
        api_keys = load_env_keys(args.env_path)
        if len(api_keys) < KEYS_PER_WORKER:
            logging.error("Need at least %s API keys; found %s", KEYS_PER_WORKER, len(api_keys))
            sys.exit(1)
        num_workers = len(api_keys) // KEYS_PER_WORKER
        logging.info("Text augmentation workers: %s", num_workers)
        unique_sentences_df = df_processed[["sentence_id", "sentence"]].drop_duplicates()
        cache_path = os.path.join(proc_dir, "sentence_variations.json")
        sentence_map: Dict[str, List[str]] = {}
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                sentence_map = json.load(f)
            logging.info("Loaded %s cached variations", len(sentence_map))
        to_process: List[Tuple[str, str]] = []
        for _, row in unique_sentences_df.iterrows():
            sid = str(row["sentence_id"])
            if sid not in sentence_map:
                to_process.append((sid, row["sentence"]))
        total_items = len(to_process)
        logging.info("Sentences remaining to process: %s", total_items)
        if total_items > 0:
            chunk_size = math.ceil(total_items / num_workers)
            chunks = [to_process[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
            futures = []
            results_aggregated: Dict[str, List[str]] = {}
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                for i in range(num_workers):
                    key_subset = api_keys[i * KEYS_PER_WORKER:(i + 1) * KEYS_PER_WORKER]
                    if i < len(chunks):
                        worker_model = WORKER_MODEL_POOL[i % len(WORKER_MODEL_POOL)] if WORKER_MODEL_POOL else DEFAULT_MODEL
                        future = executor.submit(process_batch, chunks[i], key_subset, worker_model, i + 1)
                        futures.append(future)
                for future in tqdm(as_completed(futures), total=len(futures), desc="Augmentation workers"):
                    try:
                        worker_result = future.result()
                        results_aggregated.update(worker_result)
                        sentence_map.update(worker_result)
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(sentence_map, f, indent=2)
                        logging.info("Merged chunk; total cached: %s", len(sentence_map))
                    except Exception as e:
                        logging.error("A worker failed: %s", e)
            def get_variations(sid):
                sid_str = str(sid)
                if sid_str in sentence_map:
                    return json.dumps(sentence_map[sid_str])
                return json.dumps([df_processed.loc[df_processed["sentence_id"] == sid, "sentence"].iloc[0]])
            df_processed["text_variations"] = df_processed["sentence_id"].apply(get_variations)
            df_processed.to_csv(output_meta_path, index=False)
            logging.info("Saved augmented metadata to %s", output_meta_path)

    # Optional upload to GCS
    if gcs_base:
        gcs_features = f"{gcs_base}/features"
        gcs_proc = f"{gcs_base}/proc"
        run_cmd(["gcloud", "storage", "cp", "-r", output_dir, gcs_features])
        run_cmd(["gcloud", "storage", "cp", "-r", proc_dir, gcs_proc])
        logging.info("Uploaded outputs to %s (features) and %s (proc)", gcs_features, gcs_proc)

    logging.info("Pipeline complete")


if __name__ == "__main__":
    main()
