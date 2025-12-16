"""
Text2Sign preprocessing pipeline for GCE VMs.
- Downloads dataset from GCS
- Extracts MediaPipe holistic landmarks (pose/hand/face) with lenient gates for generation
- Canonicalizes skeleton, resamples to fixed FPS, trims/pads
- Saves per-video features, metadata, global stats
- Exports plots and optionally uploads outputs back to GCS
"""
import argparse
import atexit
import csv
import json
import logging
import os
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Defaults (lenient for generative Text2Sign)
DEFAULT_DATASET_GCS_URI = "gs://ghsl-datasets/text2sign_dataset"
DEFAULT_WORKDIR = "/mnt/disks/data"
DEFAULT_OUTPUT_GCS_URI = "gs://ghsl-model-artifacts/text2sign"
DEFAULT_FACE_DOWNSAMPLE = 5
TARGET_SEQ_LEN = 96
TARGET_FPS = 30
MODEL_COMPLEXITY = 2
MIN_FRAMES = 32
MIN_VALID_RATIO = 0.30
EMA_ALPHA = 0.15
VISIBILITY_THRESHOLD = 0.35
MOTION_KEEP_THRESHOLD = 5e-4
MOTION_REJECTION_THRESHOLD = 0.0
KEEP_LOW_QUALITY_DEFAULT = True

# Landmark constants
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

mp_holistic = mp.solutions.holistic
HOLISTIC = None
FACE_INDICES: List[int] = []


def run_cmd(cmd: List[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def download_dataset(gcs_uri: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    run_cmd(["gcloud", "storage", "cp", "-r", gcs_uri, dest_dir])
    folder = os.path.basename(gcs_uri.rstrip("/"))
    path = os.path.join(dest_dir, folder)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path} after download")
    return path


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def worker_init(face_downsample: int = DEFAULT_FACE_DOWNSAMPLE):
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


def canonicalize(points_xyz: np.ndarray) -> np.ndarray:
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
    rot = np.array([[np.cos(-yaw), 0, np.sin(-yaw)], [0, 1, 0], [-np.sin(-yaw), 0, np.cos(-yaw)]], dtype=np.float32)
    pts = pts @ rot.T
    return pts.astype(np.float32)


def trim_or_pad(sequence: np.ndarray, target: int = TARGET_SEQ_LEN) -> np.ndarray:
    if sequence.shape[0] == target:
        return sequence
    if sequence.shape[0] > target:
        start = (sequence.shape[0] - target) // 2
        return sequence[start:start + target]
    pad_len = target - sequence.shape[0]
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


def resample_frames(frames: np.ndarray, orig_fps: float, target_fps: float = TARGET_FPS) -> np.ndarray:
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
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


def summarize_motion(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


@atexit.register
def _close_holistic():
    global HOLISTIC
    try:
        if HOLISTIC is not None:
            HOLISTIC.close()
    except Exception:
        pass


def process_single_video(row: Dict, video_dir: str, output_dir: str, face_downsample: int, keep_low_quality: bool):
    import os as _os

    video_file = row.get("video_file")
    video_path = _os.path.join(video_dir, video_file)
    if not _os.path.exists(video_path):
        return {"status": "missing", "video_file": video_file, "error": "File not found"}

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
            return {"status": "failed", "video_file": video_file, "error": "Cannot open video"}
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

            pose_lm = res.pose_world_landmarks or res.pose_landmarks
            if pose_lm is None:
                pose_block = [0.0] * (NUM_POSE_LANDMARKS * 4)
                pose_vis = 0.0
            else:
                pose_block = []
                for lm in pose_lm.landmark:
                    pose_block.extend([lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)])
                pose_vis = float(np.mean([getattr(lm, "visibility", 1.0) for lm in pose_lm.landmark]))

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
            coords = canonicalize(pose_xyz[:, :3])
            pose_xyz[:, :3] = coords
            frame_arr[:POSE_FEATURES] = pose_xyz.reshape(-1)

            frames.append(frame_arr)
            quality_flags.append(pose_lm is not None or res.left_hand_landmarks or res.right_hand_landmarks)
            vis_scores.append(pose_vis)

        if cap:
            cap.release()

        if not frames:
            return {"status": "failed", "video_file": video_file, "error": "No frames"}

        frames = np.stack(frames, axis=0)
        motion_vals = compute_motion_energy(frames)
        quality_mask = np.array(quality_flags, dtype=bool)
        vis_scores = np.array(vis_scores, dtype=np.float32)

        valid_idx = np.where(quality_mask)[0]
        detection_ratio = float(valid_idx.size / len(frames)) if len(frames) else 0.0
        if valid_idx.size == 0:
            return {"status": "failed", "video_file": video_file, "error": "No valid detections"}

        frames = frames[valid_idx]
        vis_scores = vis_scores[valid_idx]
        motion_vals = motion_vals[valid_idx]

        keep_mask = (vis_scores >= VISIBILITY_THRESHOLD) | (motion_vals >= MOTION_KEEP_THRESHOLD)
        if not keep_mask.any():
            keep_mask[0] = True
        frames = frames[keep_mask]
        vis_scores = vis_scores[keep_mask]
        motion_vals = motion_vals[keep_mask]

        frames = resample_frames(frames, orig_fps, TARGET_FPS)
        frames = exponential_smooth(frames, alpha=EMA_ALPHA)
        frames_tp = trim_or_pad(frames, TARGET_SEQ_LEN)

        motion_summary = summarize_motion(motion_vals)
        mean_vis = float(vis_scores.mean()) if vis_scores.size else 0.0

        status = "success"
        if (frames.shape[0] < MIN_FRAMES or detection_ratio < MIN_VALID_RATIO) and not keep_low_quality:
            status = "low_quality"
        elif (frames.shape[0] < MIN_FRAMES or detection_ratio < MIN_VALID_RATIO) and keep_low_quality:
            status = "low_quality_kept"

        base = _os.path.splitext(video_file)[0]
        npy_path = _os.path.join(output_dir, f"{base}.npy")
        np.save(npy_path, frames_tp.astype(np.float32))

        return {
            "status": status,
            "video_file": video_file,
            "feature_path": npy_path,
            "num_frames": int(frames_tp.shape[0]),
            "frame_feature_dim": int(frames_tp.shape[1]),
            "detection_ratio": detection_ratio,
            "mean_visibility": mean_vis,
            "motion_mean": float(motion_summary.get("mean", 0.0)),
            "motion_p50": float(motion_summary.get("p50", 0.0)),
            "motion_p95": float(motion_summary.get("p95", 0.0)),
            "motion_max": float(motion_summary.get("max", 0.0)),
            "sentence_id": row.get("sentence_id"),
            "sentence": row.get("sentence"),
            "sentence_text": row.get("sentence"),
            "sentence_gloss": row.get("gloss", ""),
            "orig_fps": float(orig_fps),
        }

    except Exception as exc:
        return {"status": "failed", "video_file": video_file, "error": str(exc)}

    finally:
        try:
            if cap is not None:
                cap.release()
            if local_holistic is not None:
                local_holistic.close()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text2Sign preprocessing pipeline for GCE VM")
    p.add_argument("--dataset-gcs-uri", default=DEFAULT_DATASET_GCS_URI, help="GCS URI of the Text2Sign dataset")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Local working directory")
    p.add_argument("--output-gcs-uri", default=DEFAULT_OUTPUT_GCS_URI, help="Optional GCS URI to upload features/proc")
    p.add_argument("--face-downsample", type=int, default=DEFAULT_FACE_DOWNSAMPLE, help="Face landmark downsample factor")
    p.add_argument("--keep-low-quality", action="store_true", default=KEEP_LOW_QUALITY_DEFAULT, help="Keep low-quality clips")
    p.add_argument("--max-workers", type=int, default=None, help="Override worker count")
    p.add_argument("--plot-dir", default=None, help="Directory to save plots (defaults to <workdir>/proc/plots_t2s)")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(message)s")
    gcs_base = args.output_gcs_uri.rstrip("/") if args.output_gcs_uri else None

    dataset_dir = download_dataset(args.dataset_gcs_uri, args.workdir)
    video_dir = os.path.join(dataset_dir, "videos")
    # Accept both metadata.csv and sampled_metadata.csv to match provided bundles.
    meta_candidates = ["metadata.csv", "sampled_metadata.csv"]
    meta_path = None
    for name in meta_candidates:
        candidate = os.path.join(dataset_dir, name)
        if os.path.exists(candidate):
            meta_path = candidate
            break
    if meta_path is None:
        available = ", ".join(sorted(os.listdir(dataset_dir)))
        raise FileNotFoundError(f"No metadata file found. Expected one of {meta_candidates}; available: {available}")

    output_dir = os.path.join(args.workdir, "features", "text2sign_pose")
    proc_dir = os.path.join(args.workdir, "proc")
    plot_dir = args.plot_dir or os.path.join(proc_dir, "plots_t2s")
    ensure_dirs(output_dir, proc_dir, plot_dir)
    global_stats_path = os.path.join(proc_dir, "text2sign_global_stats.npz")

    logging.info("Loading metadata: %s", meta_path)
    records: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    df_meta = pd.DataFrame(records)
    if "sentence_id" in df_meta.columns:
        df_meta["sentence_id"] = df_meta["sentence_id"].astype(str)
    logging.info("Videos: %s", len(df_meta))

    cpu_count = os.cpu_count() or 1
    num_workers = args.max_workers or max(1, cpu_count - 1)
    logging.info("Workers: %s", num_workers)

    process_func = partial(
        process_single_video,
        video_dir=video_dir,
        output_dir=output_dir,
        face_downsample=args.face_downsample,
        keep_low_quality=args.keep_low_quality,
    )

    results: List[Dict] = []
    with Pool(num_workers, initializer=worker_init, initargs=(args.face_downsample,)) as pool:
        for res in tqdm(pool.imap(process_func, df_meta.to_dict("records")), total=len(df_meta), desc="Processing"):
            results.append(res)

    processed_records: List[Dict] = []
    failed: List[Dict] = []
    missing: List[Dict] = []
    lowq: List[Dict] = []
    for r in results:
        status = r.get("status")
        if status in ("success", "low_quality_kept"):
            processed_records.append(r)
            if status == "low_quality_kept":
                lowq.append({**r, "kept": True})
        elif status == "low_quality":
            lowq.append({**r, "kept": False})
        elif status == "failed":
            failed.append(r)
        elif status == "missing":
            missing.append(r)

    df_processed = pd.DataFrame(processed_records)
    if gcs_base and not df_processed.empty:
        gcs_features = f"{gcs_base}/features"
        df_processed["feature_path_local"] = df_processed["feature_path"]
        df_processed["feature_path"] = df_processed["feature_path"].apply(
            lambda p: f"{gcs_features}/text2sign_pose/{os.path.basename(p)}"
        )
        df_processed["feature_path_gcs"] = df_processed["feature_path"]

    output_meta_path = os.path.join(proc_dir, "text2sign_processed_metadata.csv")
    df_processed.to_csv(output_meta_path, index=False)
    logging.info("Processed: %s | LowQ: %s | Failed: %s | Missing: %s", len(processed_records), len(lowq), len(failed), len(missing))
    logging.info("Saved metadata: %s", output_meta_path)

    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(proc_dir, "text2sign_failed_videos.csv"), index=False)
    if missing:
        pd.DataFrame(missing).to_csv(os.path.join(proc_dir, "text2sign_missing_videos.csv"), index=False)
    if lowq:
        pd.DataFrame(lowq).to_csv(os.path.join(proc_dir, "text2sign_low_quality.csv"), index=False)

    if not df_processed.empty:
        total_frames = 0
        sum_feats = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
        sum_sq = np.zeros(PER_FRAME_FEATURES, dtype=np.float64)
        seq_lengths: List[int] = []
        motion_means: List[float] = []
        for row in tqdm(df_processed.itertuples(), total=len(df_processed), desc="Stats"):
            arr = np.load(row.feature_path).astype(np.float32)
            total_frames += arr.shape[0]
            sum_feats += arr.sum(axis=0)
            sum_sq += (arr ** 2).sum(axis=0)
            seq_lengths.append(arr.shape[0])
            if hasattr(row, "motion_mean"):
                motion_means.append(row.motion_mean)
        feature_mean = sum_feats / max(total_frames, 1)
        feature_var = sum_sq / max(total_frames, 1) - feature_mean ** 2
        feature_std = np.sqrt(np.maximum(feature_var, 1e-8))
        np.savez(
            global_stats_path,
            feature_mean=feature_mean.astype(np.float32),
            feature_std=feature_std.astype(np.float32),
            seq_lengths=np.array(seq_lengths, dtype=np.int32),
            motion_means=np.array(motion_means, dtype=np.float32),
        )
        logging.info("Saved global stats: %s", global_stats_path)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        df_processed["num_frames"].hist(bins=30, edgecolor="black")
        plt.xlabel("Number of Frames")
        plt.ylabel("Frequency")
        plt.title("Frame Counts")
        plt.axvline(df_processed["num_frames"].mean(), color="red", linestyle="--", label="Mean")
        plt.legend()
        plt.subplot(1, 2, 2)
        if "category" in df_processed:
            cat_counts = df_processed["category"].value_counts()
            cat_counts.plot(kind="barh")
            plt.xlabel("Videos")
            plt.title("Category Distribution")
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, "frames_distribution_t2s.png")
        plt.savefig(plot_path, dpi=150)
        plt.close("all")
        logging.info("Saved plot: %s", plot_path)

    if gcs_base:
        gcs_features = f"{gcs_base}/features"
        gcs_proc = f"{gcs_base}/proc"
        run_cmd(["gcloud", "storage", "cp", "-r", output_dir, gcs_features])
        run_cmd(["gcloud", "storage", "cp", "-r", proc_dir, gcs_proc])
        logging.info("Uploaded features and proc to %s and %s", gcs_features, gcs_proc)

    logging.info("Done")


if __name__ == "__main__":
    main()
