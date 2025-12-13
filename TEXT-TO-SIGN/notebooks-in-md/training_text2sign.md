# Text2Sign — GAN-NAT Training Notebook (Markdown to copy into .ipynb)

> Implements the GAN-NAT training plan from `02-model-training.md`: non-autoregressive generator conditioned on text + style noise, 1D-CNN discriminator, reconstruction + geometric + adversarial losses. Expects preprocessing outputs from `preprocessing_text2sign.md`.

---

## Cell 0 — Dependencies (run once)

```bash
pip install torch torchvision torchaudio transformers pandas numpy tqdm
```

---

## Cell 1 — Imports & Paths

```python
import os
import math
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {DEVICE}")
```

---

## Cell 2 — Config

```python
# Data
PROCESSED_META = "proc/text2sign_processed_metadata.csv"
FEATURE_DIR = "features/text2sign_pose"
GLOBAL_STATS = "proc/text2sign_global_stats.npz"
MAX_SEQ_LEN = 128   # allow longer than preprocessing trim; we can pad
BATCH_SIZE = 16
NUM_WORKERS = 4

# Text encoder
TEXT_MODEL = "distilbert-base-uncased"
TEXT_DIM = 768
TEXT_PROJ_DIM = 256

# Generator / Discriminator
POSE_DIM = 33*4 + 2*21*3 + 94*3
LATENT_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 4
LENGTH_BINS = list(range(10, 201, 2))  # candidate lengths

# Optimization
LR_G = 1e-4
LR_D = 5e-5
BETA1, BETA2 = 0.5, 0.9
LAMBDA_REC = 1.0
LAMBDA_ADV = 1.0
LAMBDA_GEO = 0.2
GRAD_CLIP = 1.0
EPOCHS = 50
CHECKPOINT_DIR = Path("checkpoints_text2sign")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
```

---

## Cell 3 — Dataset & Dataloader

```python
class Text2SignDataset(Dataset):
    def __init__(self, meta_path, feature_dir, max_seq_len=MAX_SEQ_LEN, normalize=True):
        self.df = pd.read_csv(meta_path)
        self.feature_dir = Path(feature_dir)
        self.max_seq_len = max_seq_len
        self.normalize = normalize

        stats = np.load(GLOBAL_STATS) if normalize and os.path.exists(GLOBAL_STATS) else None
        self.mean = stats['feature_mean'] if stats is not None else None
        self.std = stats['feature_std'] if stats is not None else None

        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feat_path = Path(row['feature_path'])
        arr = np.load(feat_path)

        # Pad/trim to max_seq_len
        frames = arr
        if frames.shape[0] > self.max_seq_len:
            start = (frames.shape[0] - self.max_seq_len) // 2
            frames = frames[start:start+self.max_seq_len]
        elif frames.shape[0] < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - frames.shape[0], frames.shape[1]), dtype=frames.dtype)
            frames = np.vstack([frames, pad])

        if self.normalize and self.mean is not None:
            frames = (frames - self.mean) / (self.std + 1e-6)

        text = row.get('sentence_text') or row.get('sentence') or ""
        gloss = row.get('sentence_gloss') or ""
        full_text = gloss if gloss else text

        tokens = self.tokenizer(full_text, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        return {
            'frames': torch.tensor(frames, dtype=torch.float32),
            'text_input_ids': tokens['input_ids'].squeeze(0),
            'text_attention_mask': tokens['attention_mask'].squeeze(0),
            'seq_len': torch.tensor(min(arr.shape[0], self.max_seq_len), dtype=torch.long),
        }


def collate_fn(batch):
    frames = torch.stack([b['frames'] for b in batch])
    input_ids = torch.stack([b['text_input_ids'] for b in batch])
    attn_mask = torch.stack([b['text_attention_mask'] for b in batch])
    seq_len = torch.stack([b['seq_len'] for b in batch])
    return frames, input_ids, attn_mask, seq_len


dataset = Text2SignDataset(PROCESSED_META, FEATURE_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn, drop_last=True)
print(f"✅ Dataset ready: {len(dataset)} samples")
```

---

## Cell 4 — Text Encoder (frozen backbone + adapter)

```python
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
            seq_emb = out.last_hidden_state  # (B, T, H)
        return self.proj(seq_emb)           # (B, T, proj)
```

---

## Cell 5 — Generator (NAT) & Length Predictor

```python
class LengthPredictor(nn.Module):
    def __init__(self, in_dim=TEXT_PROJ_DIM, length_bins=LENGTH_BINS):
        super().__init__()
        self.length_bins = torch.tensor(length_bins, dtype=torch.float32)
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, len(length_bins))
        )

    def forward(self, pooled):
        logits = self.net(pooled)
        return logits


class NATBlock(nn.Module):
    def __init__(self, dim=HIDDEN_DIM, heads=4, ff_mult=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_mult*dim),
            nn.GELU(),
            nn.Linear(ff_mult*dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        # Cross-attend to text context
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
        self.input_proj = nn.Linear(hidden_dim + text_dim, hidden_dim)
        self.blocks = nn.ModuleList([NATBlock(hidden_dim, heads=4) for _ in range(layers)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pose_dim)
        )

    def forward(self, text_ctx, seq_len=None):
        B, T_txt, C = text_ctx.shape
        pooled = text_ctx[:, 0]  # use CLS/token 0
        len_logits = self.length_head(pooled)

        # sample length (teacher-forcing with provided seq_len if available)
        if seq_len is not None:
            target_len = seq_len
        else:
            probs = torch.softmax(len_logits, dim=-1)
            target_len = torch.multinomial(probs, num_samples=1).squeeze(-1)
            target_len = target_len.clamp(0, len(LENGTH_BINS)-1)
        target_len_vals = torch.tensor(LENGTH_BINS, device=text_ctx.device)[target_len]
        max_len = MAX_SEQ_LEN

        # build time queries
        z = torch.randn(B, max_len, LATENT_DIM, device=text_ctx.device)
        style = self.style_proj(z)
        t_idx = torch.arange(max_len, device=text_ctx.device).unsqueeze(0).repeat(B,1)
        time_emb = self.time_embed(t_idx)

        x = torch.cat([style, time_emb], dim=-1)
        # repeat pooled text to concatenate
        pooled_rep = pooled.unsqueeze(1).repeat(1, max_len, 1)
        x = torch.cat([x, pooled_rep], dim=-1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x, text_ctx)

        poses = self.output(x)
        return poses, len_logits, target_len_vals
```

---

## Cell 6 — Discriminator (1D CNN)

```python
class Discriminator(nn.Module):
    def __init__(self, pose_dim=POSE_DIM, base=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(pose_dim, base, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base, base, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base, base*2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base*2, base*4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(base*4, 1)

    def forward(self, x):
        # x: (B, T, pose_dim)
        x = x.transpose(1, 2)
        feat = self.net(x).squeeze(-1)
        return self.head(feat)
```

---

## Cell 7 — Losses

```python
def huber_loss(pred, target, delta=1.0):
    return F.smooth_l1_loss(pred, target, beta=delta)


def recon_losses(pred, target):
    pos = huber_loss(pred, target)
    vel = huber_loss(pred[:,1:] - pred[:,:-1], target[:,1:] - target[:,:-1])
    acc = huber_loss((pred[:,2:] - pred[:,1:-1]) - (pred[:,1:-1] - pred[:,:-2]), (target[:,2:] - target[:,1:-1]) - (target[:,1:-1] - target[:,:-2]))
    return pos + vel + acc


def gan_losses(real_logits, fake_logits):
    d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    g_loss = -fake_logits.mean()
    return d_loss, g_loss


def bone_length_consistency(poses):
    # Simple proxy: variance of pairwise bone lengths (pose coords only x,y,z)
    coords = poses.view(poses.shape[0], poses.shape[1], NUM_POSE_LANDMARKS, 4)[...,:3]
    # shoulders and hips
    ls = torch.norm(coords[:,:,LEFT_SHOULDER] - coords[:,:,RIGHT_SHOULDER], dim=-1)
    hs = torch.norm(coords[:,:,LEFT_HIP] - coords[:,:,RIGHT_HIP], dim=-1)
    spine = torch.norm(0.5*(coords[:,:,LEFT_SHOULDER]+coords[:,:,RIGHT_SHOULDER]) - 0.5*(coords[:,:,LEFT_HIP]+coords[:,:,RIGHT_HIP]), dim=-1)
    stacked = torch.stack([ls, hs, spine], dim=-1)
    return stacked.var(dim=-1).mean()
```

---

## Cell 8 — Initialize Models & Optimizers

```python
text_encoder = FrozenTextEncoder().to(DEVICE)
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(BETA1, BETA2))
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(BETA1, BETA2))

scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
```

---

## Cell 9 — Training Loop

```python
def save_ckpt(epoch):
    torch.save({
        'G': G.state_dict(),
        'D': D.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
        'epoch': epoch,
    }, CHECKPOINT_DIR / f"epoch_{epoch}.pt")


for epoch in range(1, EPOCHS + 1):
    G.train(); D.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for frames, input_ids, attn_mask, seq_len in pbar:
        frames = frames.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        seq_len = seq_len.to(DEVICE)

        with torch.no_grad():
            text_ctx = text_encoder(input_ids, attn_mask)

        # --- Train Discriminator ---
        opt_D.zero_grad()
        with torch.no_grad():
            fake, _, _ = G(text_ctx, seq_len)
        real_logits = D(frames)
        fake_logits = D(fake.detach())
        d_loss, _ = gan_losses(real_logits, fake_logits)
        d_loss.backward()
        nn.utils.clip_grad_norm_(D.parameters(), GRAD_CLIP)
        opt_D.step()

        # --- Train Generator ---
        opt_G.zero_grad()
        fake, len_logits, target_len_vals = G(text_ctx, seq_len)
        fake_logits = D(fake)
        _, g_gan = gan_losses(real_logits=None, fake_logits=fake_logits)  # g part only

        rec = recon_losses(fake, frames)
        geo = bone_length_consistency(fake)

        g_loss = LAMBDA_REC * rec + LAMBDA_ADV * g_gan + LAMBDA_GEO * geo
        g_loss.backward()
        nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
        opt_G.step()

        pbar.set_postfix({
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'rec': rec.item(),
            'geo': geo.item(),
        })

    save_ckpt(epoch)
    print(f"💾 Saved checkpoint epoch {epoch}")
```

---

## Cell 10 — Inference / Sampling

```python
@torch.no_grad()
def sample(texts, max_len=MAX_SEQ_LEN, checkpoint=None):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        G.load_state_dict(ckpt['G'])
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    text_ctx = text_encoder(tokens['input_ids'], tokens['attention_mask'])
    fake, _, _ = G(text_ctx)
    fake = fake[:, :max_len]
    return fake.cpu().numpy()

# Example
samples = sample(["Where is the hospital?", "I need a doctor"], checkpoint=None)
print(samples.shape)
```

---

## Cell 11 — Visualization: Generated Trajectory Quick Look

```python
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize_sample(text="Where is the hospital?", checkpoint=None, joint_idx=0):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=DEVICE)
        G.load_state_dict(ckpt['G'])
    tokens = AutoTokenizer.from_pretrained(TEXT_MODEL)([text], return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    text_ctx = text_encoder(tokens['input_ids'], tokens['attention_mask'])
    fake, _, _ = G(text_ctx)
    fake = fake[0].cpu().numpy()

    plt.figure(figsize=(10, 3))
    plt.plot(fake[:, joint_idx], label=f'Joint {joint_idx} (dim 0)')
    plt.title(f'Generated trajectory for "{text}"')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

visualize_sample()
```

---

## Cell 12 — Notes & Next Steps

- Tune `LAMBDA_GEO` and add richer bone sets if desired.
- Consider R1 gradient penalty if GAN training is unstable.
- Length predictor can be supervised with ground-truth lengths using cross-entropy against `seq_len` binned into `LENGTH_BINS` (add when labels are reliable).
- For better style control, feed explicit style tokens or signer IDs into the generator conditioning stream.

