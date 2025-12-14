"""
Standalone text-augmentation runner for Sign2Text.
- Loads processed_metadata.csv from GCS
- Generates text variations with Gemini
- Writes text_variations back into the metadata CSV
- Maintains/updates sentence_variations.json cache (resilient to partial JSON)
- Uploads the updated files back to the bucket
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

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# Defaults
DEFAULT_PROCESSED_META = "gs://ghsl-model-artifacts/sign2text/proc/processed_metadata.csv"
DEFAULT_OUTPUT_GCS_URI = "gs://ghsl-model-artifacts/sign2text"
DEFAULT_WORKDIR = "/mnt/disks/data"
DEFAULT_ENV_PATH = str(Path(__file__).resolve().parent.parent / ".env")
KEYS_PER_WORKER = 2
DEFAULT_MODEL = "gemini-2.0-flash"
WORKER_MODEL_POOL = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]


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


def download_from_gcs(gcs_uri: str, dest_path: str) -> None:
    dest_dir = os.path.dirname(dest_path) or "."
    ensure_dirs(dest_dir)
    run_cmd(["gcloud", "storage", "cp", gcs_uri, dest_path])


def try_download_variations(gcs_uri: str, dest_path: str) -> None:
    try:
        download_from_gcs(gcs_uri, dest_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("No existing sentence_variations.json found (or failed to download): %s", exc)


def load_variations(cache_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to parse cached variations; treating as empty. Error: %s", exc)
        # Preserve the corrupt file for inspection
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


def augment(args: argparse.Namespace) -> None:
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s [%(levelname)s] %(message)s")

    workdir = args.workdir
    proc_dir = os.path.join(workdir, "proc")
    ensure_dirs(proc_dir)

    local_meta = os.path.join(proc_dir, os.path.basename(args.processed_meta))
    local_variations = os.path.join(proc_dir, "sentence_variations.json")

    logging.info("Downloading processed metadata from %s", args.processed_meta)
    download_from_gcs(args.processed_meta, local_meta)

    gcs_variations = None
    if args.output_gcs_uri:
        gcs_base = args.output_gcs_uri.rstrip("/")
        gcs_variations = f"{gcs_base}/proc/sentence_variations.json"
        try_download_variations(gcs_variations, local_variations)

    df = pd.read_csv(local_meta)
    unique_df = df[["sentence_id", "sentence"]].drop_duplicates()
    unique_df["sentence_id"] = unique_df["sentence_id"].astype(str)
    logging.info("Unique sentences: %s", len(unique_df))

    api_keys = load_env_keys(args.env_path)
    if len(api_keys) < KEYS_PER_WORKER:
        logging.error("Need at least %s API keys; found %s", KEYS_PER_WORKER, len(api_keys))
        sys.exit(1)
    num_workers = len(api_keys) // KEYS_PER_WORKER
    logging.info("Text augmentation workers: %s", num_workers)

    existing_map = load_variations(local_variations)

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
                with open(local_variations, "w", encoding="utf-8") as f:
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

    if args.output_gcs_uri:
        logging.info("Uploading augmented metadata and cache to GCS")
        run_cmd(["gcloud", "storage", "cp", local_meta, args.processed_meta])
        if gcs_variations:
            run_cmd(["gcloud", "storage", "cp", local_variations, gcs_variations])
    logging.info("Text augmentation complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text augmentation for Sign2Text metadata")
    p.add_argument("--processed-meta", default=DEFAULT_PROCESSED_META, help="GCS URI to processed_metadata.csv")
    p.add_argument("--output-gcs-uri", default=DEFAULT_OUTPUT_GCS_URI, help="Base GCS URI for proc folder")
    p.add_argument("--workdir", default=DEFAULT_WORKDIR, help="Local working directory for downloads/cache")
    p.add_argument("--env-path", default=DEFAULT_ENV_PATH, help="Path to .env containing Gemini keys (KEY=VALUE per line)")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment(args)
