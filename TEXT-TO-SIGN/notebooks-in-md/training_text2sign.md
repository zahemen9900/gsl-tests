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
# If running in Colab, you might need to prefix with /content/
PROCESSED_META = "proc/text2sign_processed_metadata.csv"
FEATURE_DIR = "features/text2sign_pose" 
GLOBAL_STATS = "proc/text2sign_global_stats.npz"
MAX_SEQ_LEN = 128   # allow longer than preprocessing trim; we can pad
BATCH_SIZE = 64     # Increased from 16
NUM_WORKERS = 4     # Increased for faster data throughput

# Text encoder
TEXT_MODEL = "distilbert-base-uncased"
TEXT_DIM = 768
TEXT_PROJ_DIM = 512 # Increased from 256

# Generator / Discriminator
POSE_DIM = 33*4 + 2*21*3 + 94*3
LATENT_DIM = 128    # Increased from 64
HIDDEN_DIM = 512    # Increased from 256
NUM_LAYERS = 6      # Increased from 4
LENGTH_BINS = list(range(10, 201, 2))  # candidate lengths

# Pose Constants (MediaPipe indices)
NUM_POSE_LANDMARKS = 33
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# Optimization
LR_G = 2e-4         # Scaled for larger batch
LR_D = 1e-4         # Scaled for larger batch
BETA1, BETA2 = 0.5, 0.9
LAMBDA_REC = 1.0
LAMBDA_ADV = 1.0
LAMBDA_GEO = 0.2
GRAD_CLIP = 1.0
EPOCHS = 100        # Increased for deeper model
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
        
        # Fix: Resolve local path by extracting filename and joining with FEATURE_DIR
        # This handles GCS paths or absolute VM paths (/mnt/disks/data/...) in the CSV
        raw_path = row.get('feature_path_local') or row.get('feature_path')
        filename = Path(raw_path).name
        feat_path = self.feature_dir / filename
        
        if not feat_path.exists():
            # Fallback to raw path if the joined one doesn't exist (e.g. if using absolute paths)
            feat_path = Path(raw_path)
            
        try:
            arr = np.load(feat_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find feature file at {feat_path}. Check FEATURE_DIR in Config.")

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

        # Text selection: Use augmented variations if available, else fallback to standard columns
        text_variations = row.get('text_variations')
        full_text = None
        
        if isinstance(text_variations, str) and text_variations.strip().startswith('['):
            try:
                variations = json.loads(text_variations)
                if isinstance(variations, list) and len(variations) > 0:
                    choice = random.choice(variations)
                    if isinstance(choice, str):
                        full_text = choice
            except:
                pass
        
        if full_text is None:
            # Fallback to standard columns, handling potential NaN values (floats)
            for col in ['sentence_text', 'sentence', 'sentence_gloss']:
                val = row.get(col)
                if val is not None and not pd.isna(val) and str(val).strip():
                    full_text = str(val)
                    break
        
        # Final safety check: must be a non-empty string for the tokenizer
        if not isinstance(full_text, str) or not full_text.strip():
            full_text = " "

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
loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS, 
    collate_fn=collate_fn, 
    drop_last=True,
    pin_memory=True # Faster transfer to GPU
)
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
    def __init__(self, dim=HIDDEN_DIM, heads=8, ff_mult=4):
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
        # Fix: input_proj must account for style (hidden_dim) + time (hidden_dim) + text (text_dim)
        self.input_proj = nn.Linear(2 * hidden_dim + text_dim, hidden_dim)
        self.blocks = nn.ModuleList([NATBlock(hidden_dim, heads=8) for _ in range(layers)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, pose_dim)
        )

    def forward(self, text_ctx, seq_len=None):
        B, T_txt, C = text_ctx.shape
        pooled = text_ctx[:, 0]  # use CLS/token 0
        len_logits = self.length_head(pooled)

        bins = torch.tensor(LENGTH_BINS, device=text_ctx.device)

        # sample length (teacher-forcing with provided seq_len if available)
        if seq_len is not None:
            # Bin seq_len into LENGTH_BINS indices to avoid out-of-range indexing
            seq_len = seq_len.to(text_ctx.device)
            target_idx = torch.bucketize(seq_len, bins) - 1
            target_idx = target_idx.clamp(0, len(LENGTH_BINS) - 1)
        else:
            probs = torch.softmax(len_logits, dim=-1)
            target_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            target_idx = target_idx.clamp(0, len(LENGTH_BINS)-1)
        target_len_vals = bins[target_idx]
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
    def __init__(self, pose_dim=POSE_DIM, base=256): # Increased from 128
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
    if real_logits is not None:
        d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    else:
        d_loss = torch.tensor(0.0, device=fake_logits.device)
    g_loss = -fake_logits.mean()
    return d_loss, g_loss


def bone_length_consistency(poses):
    # Simple proxy: variance of pairwise bone lengths (pose coords only x,y,z)
    # Body landmarks are the first NUM_POSE_LANDMARKS*4 values
    body_poses = poses[:, :, :NUM_POSE_LANDMARKS*4].view(poses.shape[0], poses.shape[1], NUM_POSE_LANDMARKS, 4)
    coords = body_poses[..., :3] # (B, T, NUM_POSE_LANDMARKS, 3)
    
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
def save_ckpt(tag, epoch, g_loss_val):
    torch.save({
        'G': G.state_dict(),
        'D': D.state_dict(),
        'opt_G': opt_G.state_dict(),
        'opt_D': opt_D.state_dict(),
        'epoch': epoch,
        'g_loss': g_loss_val,
    }, CHECKPOINT_DIR / f"{tag}.pt")


best_g = float('inf')
history = []

for epoch in range(1, EPOCHS + 1):
    G.train(); D.train()
    # Ensure loader is defined (from Cell 3)
    if 'loader' not in globals():
        print("❌ Error: 'loader' not found. Did you run Cell 3?")
        break
    
    # Curriculum Learning Schedule
    # Phase 1: Warmup (Reconstruction only) -> Avoids "D overpowering G" early
    # Phase 2: Geometric (Rec + Geo) -> Fixes bone stretching
    # Phase 3: Adversarial (Full) -> Adds realism
    if epoch <= 15:
        curr_lambda_adv = 0.0
        curr_lambda_geo = 0.0
        phase = "Warmup (Rec Only)"
    elif epoch <= 30:
        curr_lambda_adv = 0.0
        curr_lambda_geo = LAMBDA_GEO
        phase = "Stabilization (Rec+Geo)"
    else:
        curr_lambda_adv = LAMBDA_ADV
        curr_lambda_geo = LAMBDA_GEO
        phase = "Adversarial (Full)"

    pbar = tqdm(loader, desc=f"Epoch {epoch} [{phase}]")
    total_d = 0.0
    total_g = 0.0
    total_rec = 0.0
    total_geo = 0.0
    steps = 0
    for frames, input_ids, attn_mask, seq_len in pbar:
        frames = frames.to(DEVICE, non_blocking=True)
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attn_mask = attn_mask.to(DEVICE, non_blocking=True)
        seq_len = seq_len.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            text_ctx = text_encoder(input_ids, attn_mask)

        # --- Train Discriminator (Only in Adversarial Phase) ---
        d_loss_val = 0.0
        if curr_lambda_adv > 0:
            opt_D.zero_grad()
            with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
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

        # --- Train Generator ---
        opt_G.zero_grad()
        with torch.amp.autocast('cuda', enabled=DEVICE.type == 'cuda'):
            fake, len_logits, target_len_vals = G(text_ctx, seq_len)
            
            # GAN loss (only if active)
            if curr_lambda_adv > 0:
                fake_logits = D(fake)
                _, g_gan = gan_losses(real_logits=None, fake_logits=fake_logits)
            else:
                g_gan = torch.tensor(0.0, device=DEVICE)

            rec = recon_losses(fake, frames)
            geo = bone_length_consistency(fake)

            g_loss = LAMBDA_REC * rec + curr_lambda_adv * g_gan + curr_lambda_geo * geo
        
        scaler.scale(g_loss).backward()
        scaler.unscale_(opt_G)
        nn.utils.clip_grad_norm_(G.parameters(), GRAD_CLIP)
        scaler.step(opt_G)
        scaler.update()

        pbar.set_postfix({
            'd_loss': f"{d_loss_val:.4f}",
            'g_loss': f"{g_loss.item():.4f}",
            'rec': f"{rec.item():.4f}",
            'geo': f"{geo.item():.4f}",
        })
        total_d += float(d_loss_val)
        total_g += float(g_loss.item())
        total_rec += float(rec.item())
        total_geo += float(geo.item())
        steps += 1
    # Save best (lowest g_loss) and always keep latest
    if g_loss.item() < best_g:
        best_g = g_loss.item()
        save_ckpt("best", epoch, best_g)
        print(f"💾 Saved BEST checkpoint at epoch {epoch} (g_loss={best_g:.4f})")
    save_ckpt("latest", epoch, g_loss.item())
    print(f"💾 Saved latest checkpoint epoch {epoch}")
    if steps > 0:
        history.append({
            'epoch': epoch,
            'd_loss': total_d / steps,
            'g_loss': total_g / steps,
            'rec': total_rec / steps,
            'geo': total_geo / steps,
        })
```
## Cell 10 — Quick Training Curves

```python
import matplotlib.pyplot as plt

if history:
    epochs = [h['epoch'] for h in history]
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, [h['g_loss'] for h in history], label='G Loss')
    plt.plot(epochs, [h['d_loss'] for h in history], label='D Loss')
    plt.title('GAN Losses')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(epochs, [h['rec'] for h in history], label='Reconstruction')
    plt.plot(epochs, [h['geo'] for h in history], label='Geometry')
    plt.title('Reconstruction / Geometry')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
else:
    print("No history captured; run training cell first.")
```

---

## Cell 11 — Inference / Sampling

---

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

## Cell 12 — Visualization: Generated Trajectory Quick Look

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

