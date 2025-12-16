"""
Seq2Seq Sign2Text training script for GPU VMs.
Based on notebooks-in-md/signtalk_local_training.md (contrastive + seq2seq focus).
- Encodes pose sequences; decodes text with Transformer decoder
- Uses teacher forcing cross-entropy; optional simple contrastive term with CLS pool
- Saves checkpoints and training history
"""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_PROCESSED_META = "gs://ghsl-model-artifacts/sign2text/proc/processed_metadata.csv"
DEFAULT_FEATURE_DIR = "gs://ghsl-model-artifacts/sign2text/features/pose_data"
DEFAULT_GLOBAL_STATS = "gs://ghsl-model-artifacts/sign2text/proc/global_stats.npz"
DEFAULT_OUT_DIR = "gs://ghsl-model-artifacts/sign2text/runs"
DEFAULT_GCS_CACHE = "/tmp/gcs_cache_sign2text"
MAX_SEQ_LEN = 128
TEXT_MAX_LEN = 64
BATCH_SIZE = 128  # Increased for contrastive learning
NUM_WORKERS = 4
POSE_DIM = 33 * 4 + 2 * 21 * 3 + 94 * 3
HIDDEN_DIM = 512  # Increased capacity
PROJ_DIM = 256    # Projection head dim
N_HEAD = 8        # Increased heads
NUM_LAYERS = 6    # Deeper model
LR = 3e-4
BETA1, BETA2 = 0.9, 0.98
GRAD_CLIP = 1.0
EPOCHS = 50       # More epochs
WARMUP_STEPS = 4000
CHECKPOINT_DIR = "checkpoints_sign2text"
TEXT_MODEL = "distilbert-base-uncased"
LAMBDA_CLIP = 1.0
TEMPERATURE = 0.05


# -----------------------------
# GCS helpers
# -----------------------------
def run_cmd(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")


def ensure_local(path: str, cache_root: Path) -> Path:
    if not str(path).startswith("gs://"):
        return Path(path)
    rel = path[len("gs://"):]
    local_path = cache_root / rel
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(["gcloud", "storage", "cp", path, str(local_path)])
    return local_path


def sync_dir_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    run_cmd(["gcloud", "storage", "cp", "-r", str(local_dir), gcs_uri.rstrip("/")])


# -----------------------------
# Dataset
# -----------------------------
class Sign2TextDataset(Dataset):
    def __init__(self, meta_path: str, feature_dir: str, max_seq_len: int, stats_path: str, text_max_len: int, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.feature_dir = feature_dir
        meta_local = ensure_local(meta_path, self.cache_dir)
        self.df = pd.read_csv(meta_local)
        self.max_seq_len = max_seq_len
        self.text_max_len = text_max_len
        stats_local = ensure_local(stats_path, self.cache_dir)
        stats = np.load(stats_local) if stats_local.exists() else None
        self.mean = stats["feature_mean"] if stats is not None else None
        self.std = stats["feature_std"] if stats is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = row["feature_path"]
        if not str(feat_path).startswith("gs://") and not os.path.isabs(str(feat_path)) and self.feature_dir:
            feat_path = str(Path(self.feature_dir) / feat_path)
        local_feat = ensure_local(str(feat_path), self.cache_dir)
        frames = np.load(local_feat)
        if frames.shape[0] > self.max_seq_len:
            start = (frames.shape[0] - self.max_seq_len) // 2
            frames = frames[start:start + self.max_seq_len]
        elif frames.shape[0] < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - frames.shape[0], frames.shape[1]), dtype=frames.dtype)
            frames = np.vstack([frames, pad])
        if self.mean is not None:
            frames = (frames - self.mean) / (self.std + 1e-6)

        text = row.get("sentence_text") or row.get("sentence") or ""
        toks = self.tokenizer(
            text,
            max_length=self.text_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "frames": torch.tensor(frames, dtype=torch.float32),
            "input_ids": toks["input_ids"].squeeze(0),
            "attn_mask": toks["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attn_mask = torch.stack([b["attn_mask"] for b in batch])
    return frames, input_ids, attn_mask


# -----------------------------
# Models
# -----------------------------
class FrozenTextEncoder(nn.Module):
    def __init__(self, model_name=TEXT_MODEL, proj_dim=PROJ_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(self.backbone.config.hidden_size, proj_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            # Use CLS token (index 0)
            cls_emb = out.last_hidden_state[:, 0]
        return self.proj(cls_emb)


class PoseEncoder(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, hidden_dim=HIDDEN_DIM, n_heads=N_HEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.input_proj = nn.Linear(pose_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, batch_first=True, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(MAX_SEQ_LEN, hidden_dim) * 0.02)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb[: x.size(1)]
        enc = self.encoder(x)
        pooled = enc.mean(dim=1)
        return enc, pooled


class TextDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim=HIDDEN_DIM, n_heads=N_HEAD, num_layers=NUM_LAYERS, pad_token_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_emb = nn.Parameter(torch.randn(TEXT_MAX_LEN, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, batch_first=True, dropout=0.2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt_ids, memory, tgt_mask=None):
        tgt = self.emb(tgt_ids) + self.pos_emb[: tgt_ids.size(1)]
        dec = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.fc(dec)


class Sign2TextModel(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int):
        super().__init__()
        self.encoder = PoseEncoder()
        self.decoder = TextDecoder(vocab_size=vocab_size, pad_token_id=pad_token_id)
        self.visual_proj = nn.Linear(HIDDEN_DIM, PROJ_DIM)
        self.text_encoder = FrozenTextEncoder()

    def forward(self, frames, tgt_ids, tgt_mask=None, text_input_ids=None, text_attn_mask=None):
        memory, pooled = self.encoder(frames)
        
        # Visual Projection for CLIP
        visual_proj = self.visual_proj(pooled)
        
        # Text Projection for CLIP
        text_proj = None
        if text_input_ids is not None:
            text_proj = self.text_encoder(text_input_ids, text_attn_mask)
            
        logits = self.decoder(tgt_ids, memory, tgt_mask=tgt_mask)
        return logits, visual_proj, text_proj


# -----------------------------
# Training utilities
# -----------------------------
def generate_square_subsequent_mask(sz: int, device):
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)


def label_smoothing_nll_loss(logits, target, pad_idx, eps=0.1):
    n_class = logits.size(-1)
    log_preds = F.log_softmax(logits, dim=-1)
    non_pad = target.ne(pad_idx)
    nll = -log_preds.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    smooth = -log_preds.mean(dim=-1)
    loss = (1 - eps) * nll + eps * smooth
    return (loss * non_pad).sum() / non_pad.sum()


def clip_loss(visual_emb, text_emb, temperature=TEMPERATURE):
    visual_emb = F.normalize(visual_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = (visual_emb @ text_emb.t()) / temperature
    labels = torch.arange(visual_emb.size(0), device=visual_emb.device)
    loss_v = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_v + loss_t) / 2


def save_ckpt(out_dir: Path, epoch: int, model: nn.Module, optimizer, tokenizer):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "tokenizer": tokenizer.name_or_path}, out_dir / f"epoch_{epoch}.pt")


# -----------------------------
# Training loop
# -----------------------------
def train(args):
    cache_root = Path(args.gcs_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    gcs_out = args.out_dir if str(args.out_dir).startswith("gs://") else None
    out_dir = cache_root / "runs_sign2text" if gcs_out else Path(args.out_dir)
    ckpt_dir = out_dir / CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = Sign2TextDataset(args.processed_meta, args.feature_dir, args.max_seq_len, args.global_stats, args.text_max_len, cache_dir=cache_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    tokenizer = dataset.tokenizer
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    model = Sign2TextModel(vocab_size=vocab_size, pad_token_id=pad_id).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(BETA1, BETA2), weight_decay=1e-4)

    history = []
    best_loss = math.inf

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for frames, input_ids, attn_mask in pbar:
            frames = frames.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)

            tgt_inp = input_ids[:, :-1]
            tgt_out = input_ids[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1), DEVICE)

            optimizer.zero_grad()
            
            # Forward pass with text inputs for CLIP
            logits, visual_proj, text_proj = model(frames, tgt_inp, tgt_mask, text_input_ids=input_ids, text_attn_mask=attn_mask)
            
            # Losses
            ce = label_smoothing_nll_loss(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), pad_id, eps=0.1)
            c_loss = clip_loss(visual_proj, text_proj)

            loss = ce + args.lambda_clip * c_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": loss.item(), "ce": ce.item(), "clip": c_loss.item()})

        history.append({"epoch": epoch, "loss": float(loss.item()), "ce": float(ce.item()), "clip": float(c_loss.item())})
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
        save_ckpt(ckpt_dir, epoch, model, optimizer, tokenizer)
        torch.save(model.state_dict(), out_dir / "latest.pt")
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), out_dir / "best.pt")

        if gcs_out:
            sync_dir_to_gcs(out_dir, gcs_out)

    print("Training complete. Best loss:", best_loss)


# -----------------------------
# Args
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train Seq2Seq Sign2Text")
    ap.add_argument("--processed-meta", default=DEFAULT_PROCESSED_META, help="Path to processed metadata CSV")
    ap.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR, help="Directory with pose npy files")
    ap.add_argument("--global-stats", default=DEFAULT_GLOBAL_STATS, help="Path to global stats npz")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--text-max-len", type=int, default=TEXT_MAX_LEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lambda-clip", type=float, default=LAMBDA_CLIP)
    ap.add_argument("--gcs-cache-dir", default=DEFAULT_GCS_CACHE, help="Local cache directory for GCS downloads")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
