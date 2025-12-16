"""
Standalone text augmentation + global stats recompute for Text2Sign.
- Uses LOCAL features under /mnt/disks/data/features (not GCS paths)
- Uses LOCAL metadata under /mnt/disks/data/proc/text2sign_processed_metadata.csv
- Generates Gemini text variations per sentence, caches to sentence_variations.json
- Recomputes global stats from local .npy features and writes text2sign_global_stats.npz
- Uploads augmented metadata, variations JSON, and global stats to
  gs://ghsl-model-artifacts/text2sign/proc
"""
import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# Defaults
DEFAULT_WORKDIR = "/mnt/disks/data"
DEFAULT_META_PATH = "/mnt/disks/data/proc/text2sign_processed_metadata.csv"
DEFAULT_FEATURE_DIR = "/mnt/disks/data/features/text2sign_pose"
DEFAULT_OUTPUT_GCS_URI = "gs://ghsl-model-artifacts/text2sign/proc"
DEFAULT_ENV_PATH = str(Path(__file__).resolve().parent.parent / ".env")
DEFAULT_MODEL = "gemini-2.0-flash"
WORKER_MODEL_POOL = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]
KEYS_PER_WORKER = 2


def run_cmd(cmd: List[str]) -> None:
    logging.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error("Command failed: %s", result.stderr.strip())
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    if result.stdout:
        logging.debug(result.stdout.strip())


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_env_keys(env_path: str) -> List[str]:
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


def load_variations(cache_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to parse cached variations; treating as empty. Error: %s", exc)
        corrupt_path = f"{cache_path}.corrupt"
        try:
            os.replace(cache_path, corrupt_path)
            logging.warning("Corrupt cache moved to %s", corrupt_path)
        except Exception:  # noqa: BLE001
            pass
        return {}


def process_batch(batch_data: List[Tuple[str, str]], worker_keys: List[str], worker_model: str, worker_id: int) -> Dict[str, List[str]]:
    current_key_idx = 0
    model_name = worker_model or DEFAULT_MODEL
    genai.configure(api_key=worker_keys[current_key_idx])
    local_results: Dict[str, List[str]] = {}

    def rotate_key() -> None:
        nonlocal current_key_idx
        if len(worker_keys) > 1:
            current_key_idx = (current_key_idx + 1) % len(worker_keys)
            genai.configure(api_key=worker_keys[current_key_idx])
            logging.info("[Worker %s] Rotated to Key #%s", worker_id, current_key_idx)

    def generate_variations(sentence: str, retries: int = 3) -> List[str]:
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
            except Exception as e:  # noqa: BLE001
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
        variations = generate_variations(sentence)
        if sentence not in variations:
            variations.insert(0, sentence)
        local_results[sid] = variations
        processed_count += 1
        if processed_count % 10 == 0:
            logging.info("[Worker %s] Progress: %s/%s", worker_id, processed_count, len(batch_data))
        time.sleep(1.0)
    logging.info("[Worker %s] Done chunk: %s items", worker_id, processed_count)
    return local_results


def recompute_global_stats(df: pd.DataFrame, feature_dir: str, out_path: str) -> None:
    total_frames = 0
    sum_feats = None
    sum_sq = None
    seq_lengths: List[int] = []
    motion_means: List[float] = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Global stats"):
        # Prefer local feature path; fall back to joining feature_dir with basename
        fpath = getattr(row, "feature_path", "") or ""
        local_path = fpath if os.path.exists(fpath) else os.path.join(feature_dir, os.path.basename(fpath))
        if not os.path.exists(local_path):
            continue
        arr = np.load(local_path).astype(np.float32)
        if sum_feats is None:
            sum_feats = np.zeros(arr.shape[1], dtype=np.float64)
            sum_sq = np.zeros(arr.shape[1], dtype=np.float64)
        total_frames += arr.shape[0]
        sum_feats += arr.sum(axis=0)
        sum_sq += (arr ** 2).sum(axis=0)
        seq_lengths.append(arr.shape[0])
        if hasattr(row, "motion_mean"):
            motion_means.append(float(getattr(row, "motion_mean")))

    if total_frames == 0 or sum_feats is None:
        raise RuntimeError("No frames found while computing global stats; verify feature paths.")

    feature_mean = sum_feats / total_frames
    feature_var = sum_sq / total_frames - feature_mean ** 2
    feature_std = np.sqrt(np.maximum(feature_var, 1e-8))

    np.savez(
        out_path,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        seq_lengths=np.array(seq_lengths, dtype=np.int32),
        motion_means=np.array(motion_means, dtype=np.float32),
    )
    logging.info("Saved global stats: %s", out_path)


def augment_and_stats(args: argparse.Namespace) -> None:
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(message)s")

    workdir = args.workdir
    proc_dir = os.path.join(workdir, "proc")
    ensure_dirs(proc_dir)

    local_meta = args.processed_meta
    if not os.path.exists(local_meta):
        logging.error("Processed metadata not found at %s", local_meta)
        sys.exit(1)

    variations_path = os.path.join(proc_dir, "sentence_variations.json")
    df = pd.read_csv(local_meta)
    df["sentence_id"] = df["sentence_id"].astype(str)
    unique_df = df[["sentence_id", "sentence"]].drop_duplicates()
    logging.info("Unique sentences: %s", len(unique_df))

    api_keys = load_env_keys(args.env_path)
    if len(api_keys) < KEYS_PER_WORKER:
        logging.error("Need at least %s API keys; found %s", KEYS_PER_WORKER, len(api_keys))
        sys.exit(1)
    num_workers = len(api_keys) // KEYS_PER_WORKER
    logging.info("Text augmentation workers: %s", num_workers)

    existing_map = load_variations(variations_path)

    to_process: List[Tuple[str, str]] = []
    for _, row in unique_df.iterrows():
        sid = str(row["sentence_id"])
        sentence = row["sentence"]
        if sid not in existing_map:
            to_process.append((sid, sentence))
    total_items = len(to_process)
    logging.info("Sentences remaining to process: %s", total_items)

    if total_items:
        chunk_size = math.ceil(total_items / num_workers)
        chunks = [to_process[i : i + chunk_size] for i in range(0, total_items, chunk_size)]
        futures = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(num_workers):
                key_subset = api_keys[i * KEYS_PER_WORKER : (i + 1) * KEYS_PER_WORKER]
                if i < len(chunks):
                    worker_model = WORKER_MODEL_POOL[i % len(WORKER_MODEL_POOL)] if WORKER_MODEL_POOL else DEFAULT_MODEL
                    future = executor.submit(process_batch, chunks[i], key_subset, worker_model, i + 1)
                    futures.append(future)
            for future in tqdm(as_completed(futures), total=len(futures), desc="Augmentation workers"):
                worker_result = future.result()
                existing_map.update(worker_result)
                with open(variations_path, "w", encoding="utf-8") as f:
                    json.dump(existing_map, f, indent=2)
                logging.info("Merged chunk; total cached: %s", len(existing_map))
    else:
        logging.info("All sentences already have cached variations; nothing to process.")

    def get_variations(sid: str) -> str:
        sid_str = str(sid)
        if sid_str in existing_map:
            return json.dumps(existing_map[sid_str])
        sentence_row = df.loc[df["sentence_id"].astype(str) == sid_str, "sentence"]
        if not sentence_row.empty:
            return json.dumps([sentence_row.iloc[0]])
        return json.dumps([])

    df["text_variations"] = df["sentence_id"].apply(get_variations)
    df.to_csv(local_meta, index=False)
    logging.info("Saved augmented metadata to %s", local_meta)

    # Recompute global stats from local features
    stats_path = os.path.join(proc_dir, "text2sign_global_stats.npz")
    recompute_global_stats(df, args.feature_dir, stats_path)

    # Upload artifacts to GCS
    if args.output_gcs_uri:
        gcs_proc = args.output_gcs_uri.rstrip("/")
        logging.info("Uploading metadata + variations + stats to %s", gcs_proc)
        run_cmd(["gcloud", "storage", "cp", local_meta, f"{gcs_proc}/text2sign_processed_metadata.csv"])
        run_cmd(["gcloud", "storage", "cp", variations_path, f"{gcs_proc}/sentence_variations.json"])
        run_cmd(["gcloud", "storage", "cp", stats_path, f"{gcs_proc}/text2sign_global_stats.npz"])
    logging.info("Text2Sign augmentation + stats complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text augmentation + stats for Text2Sign")
    p.add_argument("--processed-meta", default=DEFAULT_META_PATH, help="Local path to text2sign_processed_metadata.csv")
    p.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR, help="Local directory with .npy features (text2sign_pose)")
    p.add_argument("--output-gcs-uri", default=DEFAULT_OUTPUT_GCS_URI, help="GCS proc folder to upload outputs")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Local working directory (for proc/features)")
    p.add_argument("--env-path", default=DEFAULT_ENV_PATH, help="Path to .env containing Gemini keys (KEY=VALUE per line)")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment_and_stats(args)
