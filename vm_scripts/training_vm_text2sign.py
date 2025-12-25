"""
GAN-NAT Text2Sign training script for GPU VMs.
Derived from notebooks-in-md/training_text2sign.md.
- Loads preprocessed features/metadata from Text2Sign preprocessing
- Trains Generator + Discriminator with recon + adversarial + geometric losses
- Saves checkpoints and training history
"""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

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
DEFAULT_PROCESSED_META = "gs://ghsl-model-artifacts/text2sign/proc/text2sign_processed_metadata.csv"
DEFAULT_FEATURE_DIR = "gs://ghsl-model-artifacts/text2sign/features/text2sign_pose"
DEFAULT_GLOBAL_STATS = "gs://ghsl-model-artifacts/text2sign/proc/text2sign_global_stats.npz"
DEFAULT_OUT_DIR = "gs://ghsl-model-artifacts/text2sign/runs"
DEFAULT_GCS_CACHE = "/tmp/gcs_cache_text2sign"
MAX_SEQ_LEN = 128
BATCH_SIZE = 64
NUM_WORKERS = 4
TEXT_MODEL = "distilbert-base-uncased"
TEXT_PROJ_DIM = 512
POSE_DIM = 33 * 4 + 2 * 21 * 3 + 94 * 3
LATENT_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS = 6
LENGTH_BINS = list(range(10, 201, 2))
LR_G = 2e-4
LR_D = 1e-4
BETA1, BETA2 = 0.5, 0.9
LAMBDA_REC = 1.0
LAMBDA_ADV = 1.0
LAMBDA_GEO = 0.2
GRAD_CLIP = 1.0
EPOCHS = 100
CHECKPOINT_DIR = "checkpoints_text2sign"
# Landmark indices (MediaPipe) for simple bone-length consistency
NUM_POSE_LANDMARKS = 33
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


# -----------------------------
# GCS helpers
# -----------------------------
def run_cmd(cmd: List[str]) -> None:
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
class Text2SignDataset(Dataset):
    def __init__(self, meta_path: str, feature_dir: str, max_seq_len: int, stats_path: str, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.feature_dir = feature_dir
        meta_local = ensure_local(meta_path, self.cache_dir)
        self.df = pd.read_csv(meta_local)
        self.max_seq_len = max_seq_len
        stats_local = ensure_local(stats_path, self.cache_dir)
        stats = np.load(stats_local) if stats_local.exists() else None
        self.mean = stats["feature_mean"] if stats is not None else None
        self.std = stats["feature_std"] if stats is not None else None
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feat_path = row["feature_path"]
        if not str(feat_path).startswith("gs://") and not os.path.isabs(str(feat_path)) and self.feature_dir:
            feat_path = str(Path(self.feature_dir) / feat_path)
        local_feat = ensure_local(str(feat_path), self.cache_dir)
        arr = np.load(local_feat)
        frames = arr
        if frames.shape[0] > self.max_seq_len:
            start = (frames.shape[0] - self.max_seq_len) // 2
            frames = frames[start:start + self.max_seq_len]
        elif frames.shape[0] < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - frames.shape[0], frames.shape[1]), dtype=frames.dtype)
            frames = np.vstack([frames, pad])
        if self.mean is not None:
            frames = (frames - self.mean) / (self.std + 1e-6)

        text = row.get("sentence_text") or row.get("sentence") or ""
        gloss = row.get("sentence_gloss") or ""
        full_text = gloss if gloss else text
        tokens = self.tokenizer(full_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        return {
            "frames": torch.tensor(frames, dtype=torch.float32),
            "input_ids": tokens["input_ids"].squeeze(0),
            "attn_mask": tokens["attention_mask"].squeeze(0),
            "seq_len": torch.tensor(min(arr.shape[0], self.max_seq_len), dtype=torch.long),
        }


def collate_fn(batch):
    frames = torch.stack([b["frames"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attn_mask = torch.stack([b["attn_mask"] for b in batch])
    seq_len = torch.stack([b["seq_len"] for b in batch])
    return frames, input_ids, attn_mask, seq_len


# -----------------------------
# Models
# -----------------------------
class FrozenTextEncoder(nn.Module):
    def __init__(self, model_name=TEXT_MODEL, proj_dim=TEXT_PROJ_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            seq_emb = out.last_hidden_state
        return self.proj(seq_emb)


class LengthPredictor(nn.Module):
    def __init__(self, in_dim=TEXT_PROJ_DIM):
        super().__init__()
        self.len_bins = torch.tensor(LENGTH_BINS, dtype=torch.float32)
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, len(LENGTH_BINS)))

    def forward(self, pooled):
        return self.net(pooled)


class NATBlock(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, heads=4, ff_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        attn_out, _ = self.attn(x, context, context)
        x = self.norm(x + attn_out)
        x = x + self.ff(x)
        return x


class Generator(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, text_dim=TEXT_PROJ_DIM, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, layers=NUM_LAYERS):
        super().__init__()
        self.length_head = LengthPredictor(text_dim)
        self.style_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Embedding(MAX_SEQ_LEN, hidden_dim)
        # input_proj accounts for style (hidden) + time (hidden) + text (text_dim)
        self.input_proj = nn.Linear(2 * hidden_dim + text_dim, hidden_dim)
        self.blocks = nn.ModuleList([NATBlock(hidden_dim, heads=8) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, pose_dim))

    def forward(self, text_ctx, seq_len=None):
        bsz, _, _ = text_ctx.shape
        pooled = text_ctx[:, 0]
        len_logits = self.length_head(pooled)
        bins = torch.tensor(LENGTH_BINS, device=text_ctx.device)
        if seq_len is not None:
            seq_len = seq_len.to(text_ctx.device)
            target_idx = torch.bucketize(seq_len, bins) - 1
            target_idx = target_idx.clamp(0, len(LENGTH_BINS) - 1)
        else:
            probs = torch.softmax(len_logits, dim=-1)
            target_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            target_idx = target_idx.clamp(0, len(LENGTH_BINS) - 1)
        target_len_vals = bins[target_idx]

        z = torch.randn(bsz, MAX_SEQ_LEN, LATENT_DIM, device=text_ctx.device)
        style = self.style_proj(z)
        t_idx = torch.arange(MAX_SEQ_LEN, device=text_ctx.device).unsqueeze(0).repeat(bsz, 1)
        time_emb = self.time_embed(t_idx)
        x = torch.cat([style, time_emb], dim=-1)
        pooled_rep = pooled.unsqueeze(1).repeat(1, MAX_SEQ_LEN, 1)
        x = torch.cat([x, pooled_rep], dim=-1)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, text_ctx)
        poses = self.output(x)
        return poses, len_logits, target_len_vals


class Discriminator(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, base=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(pose_dim, base, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base, base, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base, base * 2, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base * 2, base * 4, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        feat = self.net(x).squeeze(-1)
        return self.head(feat)


# -----------------------------
# Losses
# -----------------------------
def huber_loss(pred, target, delta=1.0):
    return F.smooth_l1_loss(pred, target, beta=delta)


def recon_losses(pred, target):
    pos = huber_loss(pred, target)
    vel = huber_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])
    acc = huber_loss((pred[:, 2:] - pred[:, 1:-1]) - (pred[:, 1:-1] - pred[:, :-2]), (target[:, 2:] - target[:, 1:-1]) - (target[:, 1:-1] - target[:, :-2]))
    return pos + vel + acc


def gan_losses(real_logits, fake_logits):
    if real_logits is None:
        d_loss = None
    else:
        d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    g_loss = -fake_logits.mean()
    return d_loss, g_loss


def bone_length_consistency(poses):
    coords = poses.view(poses.shape[0], poses.shape[1], NUM_POSE_LANDMARKS, 4)[..., :3]
    ls = torch.norm(coords[:, :, LEFT_SHOULDER] - coords[:, :, RIGHT_SHOULDER], dim=-1)
    hs = torch.norm(coords[:, :, LEFT_HIP] - coords[:, :, RIGHT_HIP], dim=-1)
    spine = torch.norm(0.5 * (coords[:, :, LEFT_SHOULDER] + coords[:, :, RIGHT_SHOULDER]) - 0.5 * (coords[:, :, LEFT_HIP] + coords[:, :, RIGHT_HIP]), dim=-1)
    stacked = torch.stack([ls, hs, spine], dim=-1)
    return stacked.var(dim=-1).mean()


# -----------------------------
# Training
# -----------------------------
def save_ckpt(out_dir: Path, epoch: int, G: nn.Module, D: nn.Module, opt_G, opt_D):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(), "epoch": epoch}, out_dir / f"epoch_{epoch}.pt")


def train(args):
    cache_root = Path(args.gcs_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    gcs_out = args.out_dir if str(args.out_dir).startswith("gs://") else None
    out_dir = cache_root / "runs_text2sign" if gcs_out else Path(args.out_dir)
    ckpt_dir = out_dir / CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = Text2SignDataset(args.processed_meta, args.feature_dir, args.max_seq_len, args.global_stats, cache_dir=cache_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True)

    text_encoder = FrozenTextEncoder().to(DEVICE)
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(BETA1, BETA2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(BETA1, BETA2))

    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")

    history = []
    best_g_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        # Curriculum phases
        if epoch <= 15:
            curr_lambda_adv = 0.0
            curr_lambda_geo = 0.0
        elif epoch <= 30:
            curr_lambda_adv = 0.0
            curr_lambda_geo = args.lambda_geo
        else:
            curr_lambda_adv = args.lambda_adv
            curr_lambda_geo = args.lambda_geo

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        total_d = total_g = total_rec = total_geo = 0.0
        steps = 0
        for frames, input_ids, attn_mask, seq_len in pbar:
            frames = frames.to(DEVICE, non_blocking=True)
            input_ids = input_ids.to(DEVICE, non_blocking=True)
            attn_mask = attn_mask.to(DEVICE, non_blocking=True)
            seq_len = seq_len.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                text_ctx = text_encoder(input_ids, attn_mask)

            # Train D (only if adv active)
            d_loss_val = 0.0
            if curr_lambda_adv > 0:
                opt_D.zero_grad()
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    with torch.no_grad():
                        fake, _, _ = G(text_ctx, seq_len)
                    real_logits = D(frames)
                    fake_logits = D(fake.detach())
                    d_loss, _ = gan_losses(real_logits, fake_logits)
                scaler.scale(d_loss).backward()
                scaler.unscale_(opt_D)
                nn.utils.clip_grad_norm_(D.parameters(), GRAD_CLIP)
                scaler.step(opt_D)
                scaler.update()
                d_loss_val = d_loss.item()

            # Train G
            opt_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                fake, len_logits, target_len_vals = G(text_ctx, seq_len)
                if curr_lambda_adv > 0:
                    fake_logits = D(fake)
                    _, g_gan = gan_losses(real_logits=None, fake_logits=fake_logits)
                else:
                    g_gan = torch.tensor(0.0, device=DEVICE)
                rec = recon_losses(fake, frames)
                geo = bone_length_consistency(fake)
                g_loss = args.lambda_rec * rec + curr_lambda_adv * g_gan + curr_lambda_geo * geo

            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
            scaler.step(opt_G)
            scaler.update()

            pbar.set_postfix({"d_loss": d_loss_val, "g_loss": float(g_loss), "rec": float(rec), "geo": float(geo)})
            total_d += d_loss_val
            total_g += float(g_loss)
            total_rec += float(rec)
            total_geo += float(geo)
            steps += 1

        if steps > 0:
            epoch_d = total_d / steps
            epoch_g = total_g / steps
            epoch_rec = total_rec / steps
            epoch_geo = total_geo / steps
            history.append({"epoch": epoch, "d_loss": epoch_d, "g_loss": epoch_g, "rec": epoch_rec, "geo": epoch_geo})
        else:
            epoch_g = math.inf
        # Save best and latest
        if epoch_g < best_g_loss:
            best_g_loss = epoch_g
            torch.save(G.state_dict(), out_dir / "best_generator.pt")
            torch.save(D.state_dict(), out_dir / "best_discriminator.pt")
            save_ckpt(ckpt_dir, epoch, G, D, opt_G, opt_D)
        torch.save(G.state_dict(), out_dir / "generator_latest.pt")
        torch.save(D.state_dict(), out_dir / "discriminator_latest.pt")
        pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

        if gcs_out:
            sync_dir_to_gcs(out_dir, gcs_out)

    print("Training complete. Best G loss:", best_g_loss)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train GAN-NAT Text2Sign")
    ap.add_argument("--processed-meta", default=DEFAULT_PROCESSED_META, help="Path to processed metadata CSV")
    ap.add_argument("--feature-dir", default=DEFAULT_FEATURE_DIR, help="Directory with .npy features")
    ap.add_argument("--global-stats", default=DEFAULT_GLOBAL_STATS, help="Path to global stats npz")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for checkpoints and history")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--lr-g", type=float, default=LR_G)
    ap.add_argument("--lr-d", type=float, default=LR_D)
    ap.add_argument("--lambda-rec", type=float, default=LAMBDA_REC)
    ap.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV)
    ap.add_argument("--lambda-geo", type=float, default=LAMBDA_GEO)
    ap.add_argument("--gcs-cache-dir", default=DEFAULT_GCS_CACHE, help="Local cache directory for GCS downloads")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
