# SignTalk-GH — Corrected Training & Evaluation Notebook

**Improvements in this version:**
- ✅ **Dual-View Generation** (Even/Odd frames) for robust self-supervision
- ✅ **Class Balanced Sampling** for effective contrastive learning
- ✅ **Text Augmentation** integration (synonyms)
- ✅ Robust train/val split (handles single-sample sentences)
- ✅ Feature dimension validation
- ✅ Better error handling
- ✅ Efficient inference with cached embeddings
- ✅ Progress tracking and debugging utilities
- ✅ Memory-efficient embedding computation
- ✅ Cosine LR schedule with warm restarts and grad accumulation

---

## CELL 0 — Quick Notes

```markdown
Purpose: Train temporal embedding model on MediaPipe .npy features
Assumes:
 - features/pose_data/*.npy exist (540 features/frame from preprocessing)
 - processed_metadata.csv with columns: video_file, feature_path, num_frames, sentence_id, sentence, category
 - conda env with PyTorch + numpy/pandas
GPU: RTX 3050 (4 GB) -> small batches, mixed precision
```

---

## CELL 1 — Install/Verify Dependencies

```bash
!pip install torch torchvision --upgrade
!pip install scikit-learn tqdm pandas numpy
```

---

## CELL 2 — Imports & Config

```python
import os
import math
import time
import random
import json
import warnings
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Paths
PROC_META = "processed_metadata.csv"
FEATURE_DIR = "./features/pose_data"
OUT_DIR = "./runs"
os.makedirs(OUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## CELL 3 — Hyperparameters

```python
CFG = {
    "seq_len": 64,              # target frame length
    "input_dim": 540,           # from MediaPipe preprocessing (33*4 + 42*3 + 94*3)
    "proj_dim": 128,            # frame projector output
    "rnn_hidden": 128,
    "rnn_layers": 2,
    "embed_dim": 256,
    "batch_size": 6,            # tuned for 4GB GPU
    "epochs": 100,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "temperature": 0.05,        # InfoNCE temperature
    "val_split": 0.15,          # validation split ratio
    "patience": 12,             # early stopping patience
    "num_workers": 2,
    "grad_accum_steps": 2,      # gradient accumulation for larger effective batch
    "scheduler_t0": 20,         # cosine warm restart period
    "scheduler_tmult": 1,       # restart multiplier
    "scheduler_eta_min_scale": 0.1,  # eta_min = lr * scale
    "seed": SEED,
    "device": device
}

print("📋 Configuration:")
for k, v in CFG.items():
    print(f"   {k}: {v}")
```

---

## CELL 4 — Load & Validate Metadata

```python
print("\n📊 Loading metadata...")
meta = pd.read_csv(PROC_META)
print(f"   Total samples: {len(meta)}")
print(f"   Columns: {meta.columns.tolist()}")

# Validate required columns
required_cols = ['video_file', 'feature_path', 'sentence_id', 'num_frames']
missing = [col for col in required_cols if col not in meta.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Check for missing files
print("\n🔍 Validating feature files...")
missing_files = []
for idx, row in meta.iterrows():
    if not os.path.exists(row['feature_path']):
        missing_files.append(row['feature_path'])

if missing_files:
    print(f"⚠️  Warning: {len(missing_files)} files not found")
    meta = meta[meta['feature_path'].apply(os.path.exists)]
    print(f"   Proceeding with {len(meta)} valid samples")

# Sentence distribution
sentence_counts = meta['sentence_id'].value_counts()
print(f"\n📈 Dataset statistics:")
print(f"   Unique sentences: {meta['sentence_id'].nunique()}")
print(f"   Videos per sentence (avg): {len(meta) / meta['sentence_id'].nunique():.2f}")
print(f"   Min videos per sentence: {sentence_counts.min()}")
print(f"   Max videos per sentence: {sentence_counts.max()}")

meta.head()
```

---

## CELL 5 — Augmentation Functions

```python
def normalize_seq(seq):
    """Normalize features by mean/std per dimension"""
    mu = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mu) / std

def random_time_crop(seq, target_len):
    """Random crop to target length"""
    T = seq.shape[0]
    if T <= target_len:
        return seq
    start = random.randint(0, T - target_len)
    return seq[start:start+target_len]

def random_time_warp(seq, max_warp=0.15):
    """Time warping via resampling"""
    T = seq.shape[0]
    factor = 1.0 + random.uniform(-max_warp, max_warp)
    new_T = max(1, int(T * factor))
    
    idxs = np.linspace(0, T-1, new_T)
    resampled = np.zeros((new_T, seq.shape[1]), dtype=np.float32)
    
    for i, x in enumerate(idxs):
        lo = int(np.floor(x))
        hi = min(T-1, lo+1)
        w = x - lo
        resampled[i] = (1-w)*seq[lo] + w*seq[hi]
    
    return resampled

def add_noise(seq, sigma=0.01):
    """Add Gaussian noise"""
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)

def random_rotation(seq, max_angle=15):
    """Apply random Y-axis rotation to 3D coordinates"""
    # ... (See notebook for implementation) ...
    return out

print("✅ Augmentation functions defined")
```

---

## CELL 6 — Dataset Class

```python
class SignDataset(Dataset):
    """
    Returns two augmented views (Even/Odd frames) for contrastive learning
    """
    def __init__(self, meta_df, seq_len=64, expected_dim=540, augment=True, cfg=None):
        # ... (See notebook for full implementation) ...
        # Parses 'text_variations' from metadata
        pass
        
    def __getitem__(self, idx):
        # ...
        # Dual-View Splitting:
        # View 1: Even frames + Noise
        # View 2: Odd frames + Rotation
        # Returns: {'a': view1, 'b': view2, 'text_a': ..., 'text_b': ...}
        pass

def collate_fn(batch):
    """Stack batch tensors"""
    # ...
    # Returns: a, b, labels, sentence_ids, text_a, text_b
    pass

print("✅ Dataset class defined")
```

---

## CELL 7 — Smart Train/Val Split

```python
print("\n🔀 Creating train/val split...")

# Group by sentence_id to ensure no data leakage
# (same sentence renditions stay together in train or val)
sentence_groups = meta.groupby('sentence_id').size()

print(f"   Sentences with 1 sample: {(sentence_groups == 1).sum()}")
print(f"   Sentences with 2+ samples: {(sentence_groups >= 2).sum()}")

# Use GroupShuffleSplit to keep sentence groups together
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=CFG['val_split'],
    random_state=SEED
)

train_idx, val_idx = next(splitter.split(meta, groups=meta['sentence_id']))
train_meta = meta.iloc[train_idx].reset_index(drop=True)
val_meta = meta.iloc[val_idx].reset_index(drop=True)

print(f"\n📊 Split results:")
print(f"   Train: {len(train_meta)} videos, {train_meta['sentence_id'].nunique()} sentences")
print(f"   Val: {len(val_meta)} videos, {val_meta['sentence_id'].nunique()} sentences")

# Verify no sentence overlap
train_sentences = set(train_meta['sentence_id'])
val_sentences = set(val_meta['sentence_id'])
overlap = train_sentences & val_sentences

if overlap:
    print(f"⚠️  Warning: {len(overlap)} sentences appear in both train and val!")
else:
    print("✅ No sentence overlap between train and val")

# Create datasets
train_ds = SignDataset(train_meta, seq_len=CFG['seq_len'], expected_dim=CFG['input_dim'], augment=True)
val_ds = SignDataset(val_meta, seq_len=CFG['seq_len'], expected_dim=CFG['input_dim'], augment=True)

# Use ClassBalancedSampler for training
# Ensures each batch contains at least 2 instances of the same sentence
train_sampler = ClassBalancedSampler(train_meta, CFG['batch_size'], instances_per_class=2)

train_loader = DataLoader(
    train_ds,
    batch_sampler=train_sampler, # Use batch_sampler
    num_workers=CFG['num_workers'],
    collate_fn=collate_fn,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=CFG['batch_size'],
    shuffle=False,
    num_workers=CFG['num_workers'],
    collate_fn=collate_fn,
    pin_memory=True
)

print(f"\n✅ DataLoaders ready (with ClassBalancedSampler)")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")
```

---

## CELL 8 — Model Architecture

```python
class FrameProjector(nn.Module):
    """Project per-frame features to lower dimension"""
    def __init__(self, in_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU()
        )
    
    def forward(self, x):  # (B, T, F)
        B, T, F = x.shape
        x = x.reshape(B * T, F)
        out = self.net(x)
        return out.reshape(B, T, -1)

class TemporalEncoder(nn.Module):
    """Encode temporal sequence to fixed embedding"""
    def __init__(self, in_dim, hidden=128, n_layers=2, embed_dim=256):
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden * 2, embed_dim)
    
    def forward(self, x):  # (B, T, C)
        out, _ = self.gru(x)  # (B, T, 2*hidden)
        emb = out.mean(dim=1)  # Mean pooling
        emb = self.fc(emb)
        emb = nn.functional.normalize(emb, dim=-1)  # L2 normalize
        return emb

class SignEmbeddingModel(nn.Module):
    """Complete embedding model"""
    def __init__(self, input_dim, proj_dim=128, hidden=128, n_layers=2, embed_dim=256):
        super().__init__()
        self.projector = FrameProjector(input_dim, proj_dim)
        self.encoder = TemporalEncoder(proj_dim, hidden, n_layers, embed_dim)
    
    def forward(self, x):  # (B, T, F)
        x_proj = self.projector(x)
        emb = self.encoder(x_proj)
        return emb

# Instantiate model
model = SignEmbeddingModel(
    input_dim=CFG['input_dim'],
    proj_dim=CFG['proj_dim'],
    hidden=CFG['rnn_hidden'],
    n_layers=CFG['rnn_layers'],
    embed_dim=CFG['embed_dim']
).to(CFG['device'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n🤖 Model initialized")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (fp32)")
```

---

## CELL 9 — Loss & Metrics

```python
def info_nce_loss(emb_a, emb_b, temperature=0.07):
    """
    InfoNCE contrastive loss
    emb_a, emb_b: (B, d) L2-normalized embeddings
    """
    B = emb_a.shape[0]
    
    # Similarity matrix (B x B)
    logits = torch.matmul(emb_a, emb_b.T) / temperature
    labels = torch.arange(B, device=emb_a.device)
    
    # Symmetric loss
    loss_a = nn.functional.cross_entropy(logits, labels)
    loss_b = nn.functional.cross_entropy(logits.T, labels)
    
    return 0.5 * (loss_a + loss_b)

@torch.no_grad()
def compute_embeddings(model, loader, device):
    """Compute embeddings for entire dataset"""
    model.eval()
    all_embs = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Computing embeddings", leave=False):
        if batch[0] is None:  # Skip failed batches
            continue
        
        # Unpack new batch structure
        a, b, labels, *_ = batch
        
        x = a.to(device, non_blocking=True)
        emb = model(x)
        
        all_embs.append(emb.cpu())
        all_labels.append(labels)
    
    if len(all_embs) == 0:
        return None, None
    
    return torch.cat(all_embs), torch.cat(all_labels)

@torch.no_grad()
def evaluate_alignment(model, loader, device):
    """Measure how well paired augmentations align in embedding space."""
    model.eval()

    emb_a_chunks = []
    emb_b_chunks = []
    counts = 0

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        if batch[0] is None:
            continue

        # Unpack new batch structure
        a, b, labels, *_ = batch
        
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)

        emb_a_chunks.append(model(a).cpu())
        emb_b_chunks.append(model(b).cpu())
        counts += a.size(0)

    if counts == 0:
        return {"inst_top1": 0.0, "pos_sim": 0.0, "neg_sim": 0.0}

    emb_a = torch.cat(emb_a_chunks)
    emb_b = torch.cat(emb_b_chunks)
    sims = emb_a @ emb_b.T

    top1 = (sims.argmax(dim=1) == torch.arange(sims.size(0))).float().mean().item()
    pos_sim = sims.diag().mean().item()

    if sims.size(0) > 1:
        mask = torch.eye(sims.size(0), dtype=torch.bool)
        hard_neg = sims.masked_fill(mask, -1.0).max(dim=1).values.mean().item()
    else:
        hard_neg = 0.0

    return {"inst_top1": top1, "pos_sim": pos_sim, "neg_sim": hard_neg}

print("✅ Loss and metrics defined")
```

---

## CELL 10 — Training Loop

```python
def train_epoch(model, loader, optimizer, scaler, device, temperature, use_amp, grad_accum_steps=1):
    """Single training epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    step_count = 0
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        if batch[0] is None:  # Skip failed batches
            continue
        
        # Unpack new batch structure: a, b, labels, sentence_ids, text_a, text_b
        a, b, *_ = batch
        a = a.to(device, non_blocking=True)
        b = b.to(device, non_blocking=True)
        
        amp_context = autocast if use_amp else nullcontext
        with amp_context():
            emb_a = model(a)
            emb_b = model(b)
            loss = info_nce_loss(emb_a, emb_b, temperature)

        loss_to_backprop = loss / grad_accum_steps

        if scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        step_count += 1

        if step_count % grad_accum_steps == 0:
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Handle leftover gradients
    if step_count % grad_accum_steps != 0:
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return total_loss / max(num_batches, 1)

def train_model(model, train_loader, val_loader, cfg, out_dir):
    """Full training loop with early stopping"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay']
    )
    eta_min = cfg['lr'] * cfg['scheduler_eta_min_scale']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg['scheduler_t0'],
        T_mult=cfg['scheduler_tmult'],
        eta_min=eta_min
    )
    
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    best_top1 = 0.0
    epochs_no_improve = 0
    history = []
    
    print(f"\n🚀 Starting training for {cfg['epochs']} epochs")
    print("=" * 70)
    
    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler,
            cfg['device'], cfg['temperature'], use_amp,
            grad_accum_steps=cfg['grad_accum_steps']
        )
        
        # Validate
        metrics = evaluate_alignment(model, val_loader, cfg['device'])
        top1 = metrics['inst_top1']
        pos_sim = metrics['pos_sim']
        neg_sim = metrics['neg_sim']
        
        t1 = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        print(f"Epoch {epoch:02d}/{cfg['epochs']} | "
              f"Loss: {train_loss:.4f} | "
              f"Inst@1: {top1:.3f} | "
              f"PosSim: {pos_sim:.3f} | NegSim: {neg_sim:.3f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {t1-t0:.1f}s")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_inst_top1': top1,
            'val_pos_sim': pos_sim,
            'val_neg_sim': neg_sim,
            'time': t1 - t0,
            'lr': current_lr
        })

        scheduler.step(epoch)
        

        > Note: Export keeps the sequence length fixed at `CFG['seq_len']` (64 frames). Adjust the preprocessing if you plan to support variable-length sequences in ONNX.

        # Checkpointing
        if top1 > best_top1:
            best_top1 = top1
            epochs_no_improve = 0
            
            ckpt_path = os.path.join(out_dir, f"best_model_top1_{top1:.3f}.pt")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': cfg,
                'best_top1': best_top1
            }, ckpt_path)
            print(f"   ✅ Saved checkpoint: {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg['patience']:
                print(f"\n⏹️  Early stopping triggered after {epoch} epochs")
                break
    
    # Save history
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)
    
    print("\n" + "=" * 70)
    print(f"✅ Training complete! Best Inst@1: {best_top1:.3f}")
    
    return model, history

# Run training
trained_model, history = train_model(model, train_loader, val_loader, CFG, OUT_DIR)
```

---

## CELL 11 — Generate & Save Embeddings

```python
print("\n💾 Generating embeddings for all data...")

# Combine train + val for embedding database
all_meta = pd.concat([train_meta, val_meta]).reset_index(drop=True)
all_ds = SignDataset(
    all_meta,
    seq_len=CFG['seq_len'],
    expected_dim=CFG['input_dim'],
    augment=False
)
all_loader = DataLoader(
    all_ds,
    batch_size=CFG['batch_size'],
    shuffle=False,
    num_workers=CFG['num_workers'],
    collate_fn=collate_fn
)

# Compute embeddings
embs, labels = compute_embeddings(trained_model, all_loader, CFG['device'])

if embs is not None:
    embs_np = embs.numpy()
    
    # Save
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embs_np)
    
    # Save mapping
    mapping_df = all_meta[['video_file', 'sentence_id', 'sentence', 'category']].copy()
    mapping_df.to_csv(os.path.join(OUT_DIR, "embeddings_map.csv"), index=False)
    
    print(f"✅ Saved {len(embs_np)} embeddings")
    print(f"   Shape: {embs_np.shape}")
    print(f"   Files: embeddings.npy, embeddings_map.csv")
else:
    print("❌ Failed to compute embeddings")
```

---

## CELL 12 — Inference Helper (Cached)

```python
class InferenceEngine:
    """Efficient inference with cached embeddings"""
    def __init__(self, model, embeddings_path, mapping_path, cfg):
        self.model = model.eval().to(cfg['device'])
        self.cfg = cfg
        
        # Load embeddings and mapping
        self.embs = np.load(embeddings_path)
        self.mapping = pd.read_csv(mapping_path)
        
        # Normalize embeddings
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True)
        self.embs = self.embs / (norms + 1e-8)
        
        print(f"✅ Loaded {len(self.embs)} reference embeddings")
    
    def preprocess_sequence(self, feat_np):
        """Preprocess raw features"""
        seq = feat_np.astype(np.float32)

        if seq.shape[0] > self.cfg['seq_len']:
            start = (seq.shape[0] - self.cfg['seq_len']) // 2
            seq = seq[start:start + self.cfg['seq_len']]

        seq = normalize_seq(seq)

        if seq.shape[0] < self.cfg['seq_len']:
            pad = np.zeros((self.cfg['seq_len'] - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])

        if seq.shape[0] > self.cfg['seq_len']:
            seq = seq[:self.cfg['seq_len']]

        return seq
    
    @torch.no_grad()
    def predict(self, feat_np, topk=5):
        """
        Get top-k predictions for a sequence
        
        Args:
            feat_np: numpy array (num_frames, 540)
            topk: number of results to return
        
        Returns:
            list of dicts with video_file, sentence_id, sentence, similarity
        """
        # Preprocess
        x = self.preprocess_sequence(feat_np.astype(np.float32))
        x_t = torch.from_numpy(x).unsqueeze(0).to(self.cfg['device'])
        
        # Get embedding
        emb = self.model(x_t).cpu().numpy()[0]
        
        # Normalize
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        
        # Compute similarities
        sims = self.embs @ emb
        
        # Get top-k
        top_idx = sims.argsort()[::-1][:topk]
        
        results = []
        for idx in top_idx:
            results.append({
                'video_file': self.mapping.loc[idx, 'video_file'],
                'sentence_id': int(self.mapping.loc[idx, 'sentence_id']),
                'sentence': self.mapping.loc[idx, 'sentence'],
                'category': self.mapping.loc[idx, 'category'],
                'similarity': float(sims[idx]),
                'confidence': float((sims[idx] + 1) / 2)  # Scale to [0,1]
            })
        
        return results

# Initialize inference engine
engine = InferenceEngine(
    trained_model,
    os.path.join(OUT_DIR, "embeddings.npy"),
    os.path.join(OUT_DIR, "embeddings_map.csv"),
    CFG
)

print("\n✅ Inference engine ready")

# Test on a validation sample
test_row = val_meta.sample(1).iloc[0]
test_feat = np.load(test_row['feature_path'])
predictions = engine.predict(test_feat, topk=5)

print(f"\n🧪 Test prediction for: {test_row['video_file']}")
print(f"   True sentence: {test_row['sentence']}")
print(f"\n   Top 5 predictions:")
for i, pred in enumerate(predictions, 1):
    print(f"   {i}. [{pred['confidence']:.2%}] {pred['sentence'][:60]}...")
```

---

## CELL 13 — Export to ONNX

> The ONNX export assumes each clip is padded/truncated to `CFG['seq_len']` (64 frames) and bakes in the batch size used here (`1`). Keep preprocessing aligned or refactor the model before enabling variable-length or variable-batch exports.

```python
print("\n📦 Exporting model to ONNX...")

# Prepare model for export
model_cpu = trained_model.cpu().eval()

# Dummy input (fixed 64-frame sequence expected by the model)
dummy_input = torch.randn(1, CFG['seq_len'], CFG['input_dim'])

# Export (fixed-length sequence; ONNX graph is traced with batch size 1)
onnx_path = os.path.join(OUT_DIR, "sign_encoder.onnx")

torch.onnx.export(
    model_cpu,
    dummy_input,
    onnx_path,
    input_names=['sequence'],
    output_names=['embedding'],
    opset_version=14,
    do_constant_folding=True
)

print(f"✅ Exported to: {onnx_path}")
print(f"   Size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")

# Verify ONNX model
try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model verified")
except ImportError:
    print("⚠️  Install onnx package to verify: pip install onnx")
except Exception as e:
    print(f"⚠️  ONNX verification failed: {e}")
```

---

## CELL 14 — Visualize Training History

```python
import matplotlib.pyplot as plt

history_df = pd.DataFrame(history)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
axes[0].plot(history_df['epoch'], history_df['train_loss'], marker='o', label='Train Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Alignment metrics plot
axes[1].plot(history_df['epoch'], history_df['val_inst_top1'], marker='o', label='Inst@1')
axes[1].plot(history_df['epoch'], history_df['val_pos_sim'], marker='s', label='Pos Cos Sim')
axes[1].plot(history_df['epoch'], history_df['val_neg_sim'], marker='^', label='Neg Cos Sim')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Alignment Metric')
axes[1].set_title('Validation Alignment Metrics')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=150)
plt.show()

print(f"✅ Saved training curves to {OUT_DIR}/training_curves.png")
```

---

## CELL 15 — Final Summary

```python
print("\n" + "=" * 70)
print("📊 TRAINING SUMMARY")
print("=" * 70)

print(f"\n📁 Output directory: {OUT_DIR}")
print(f"\n📈 Best performance:")
print(f"   Inst@1: {max(h['val_inst_top1'] for h in history):.3f}")
print(f"   Pos Cos Sim: {max(h['val_pos_sim'] for h in history):.3f}")
print(f"   Neg Cos Sim (min): {min(h['val_neg_sim'] for h in history):.3f}")

print(f"\n📦 Generated files:")
files = [
    "best_model_top1_*.pt",
    "embeddings.npy",
    "embeddings_map.csv",
    "sign_encoder.onnx",
    "training_history.csv",
    "training_curves.png"
]
for f in files:
    print(f"   ✓ {f}")

print(f"\n🚀 Next steps:")
print(f"   1. Test inference with InferenceEngine")
print(f"   2. Upload embeddings to Supabase (pgvector)")
print(f"   3. Deploy ONNX model with FastAPI")
print(f"   4. Integrate with frontend")

print("\n" + "=" * 70)
```

---

## 🎯 Key Improvements Summary

1. **✅ Robust data split** - Uses GroupShuffleSplit to prevent sentence leakage
2. **✅ Feature validation** - Checks dimensions match expected 540
3. **✅ Error handling** - Graceful handling of corrupted files
4. **✅ Efficient inference** - InferenceEngine caches embeddings
5. **✅ Better logging** - Clear progress indicators and summaries
6. **✅ Memory efficient** - Processes embeddings in batches
7. **✅ Production-ready** - ONNX export with verification
8. **✅ Visualization** - Training curves saved automatically

---

## 💡 Usage Tips

- **Adjust batch_size** if you hit OOM (try 4 or even 2)
- **Monitor GPU usage** with `nvidia-smi` in another terminal
- **Expected training time**: ~2-3 hours for 1200 videos, 30 epochs
- **Inst@1 target**: Aim for >0.70 on validation set
- **If training is slow**: Reduce seq_len to 48 or use fewer GRU layers
- **Tune grad_accum_steps** to increase negatives per step when memory allows
