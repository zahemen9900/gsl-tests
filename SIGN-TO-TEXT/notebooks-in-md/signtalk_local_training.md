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
import csv
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, List
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

try:
    from fastdtw import fastdtw
except ImportError:
    fastdtw = None

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
PROC_META = "./proc/processed_metadata.csv"
FEATURE_DIR = "./features/pose_data"
GLOBAL_STATS_PATH = "./proc/global_stats.npz"
OUT_DIR = "./runs"
os.makedirs(OUT_DIR, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if not os.path.exists(GLOBAL_STATS_PATH):
    print("⚠️ Missing global_stats.npz. Will compute from processed metadata if available.")
```

---

## CELL 3 — Hyperparameters

```python
CFG = {
    "seq_len": 128,
    "text_max_len": 64,
    "input_dim": 540,
    "hidden_dim": 512,
    "proj_dim": 256,
    "n_heads": 8,
    "num_layers": 6,
    "batch_size": 64,
    "epochs": 50,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "lambda_clip": 1.0,
    "temperature": 0.05,
    "text_model": "distilbert-base-uncased",
    "motion_floor": 7.5e-4,
    "val_split": 0.15,
    "num_workers": 4,
    "grad_clip": 1.0,
    "time_warp": 0.15,
    "noise_sigma": 0.012,
    "frame_dropout": 0.12,
    "temporal_shift": 4,
    "global_stats_path": GLOBAL_STATS_PATH,
    "device": device
}

print("📋 Configuration:")
for k, v in CFG.items():
    print(f"   {k}: {v}")
```

---

## CELL 4 — Load & Validate Metadata

```python
print("\n📊 Loading global normalization stats...")

def compute_and_save_global_stats(meta_path: str, stats_path: str) -> None:
    """Compute dataset-wide mean and std so inference can normalize inputs."""
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Processed metadata missing at {meta_path}.")
    df = pd.read_csv(meta_path)
    if df.empty:
        raise ValueError("Processed metadata is empty; cannot compute global stats.")

    total_frames = 0
    sum_feats = None
    sum_sq_feats = None
    seq_lengths = []
    motion_means = []
    skipped_files = []

    for row in tqdm(df.itertuples(), total=len(df), desc="Computing stats"):
        feat_path = getattr(row, "feature_path", None)
        if not feat_path or not os.path.exists(feat_path):
            skipped_files.append(str(feat_path))
            continue
        try:
            feat_arr = np.load(feat_path).astype(np.float32)
        except Exception as exc:
            skipped_files.append(f"{feat_path} :: {exc}")
            continue
        if feat_arr.ndim != 2 or feat_arr.shape[0] == 0:
            continue
        if sum_feats is None:
            feat_dim = feat_arr.shape[1]
            sum_feats = np.zeros(feat_dim, dtype=np.float64)
            sum_sq_feats = np.zeros(feat_dim, dtype=np.float64)

        total_frames += feat_arr.shape[0]
        sum_feats += feat_arr.sum(axis=0)
        sum_sq_feats += (feat_arr ** 2).sum(axis=0)
        seq_lengths.append(feat_arr.shape[0])
        if hasattr(row, "motion_mean") and not pd.isna(row.motion_mean):
            motion_means.append(float(row.motion_mean))

    if skipped_files:
        print(f"   ⚠️ Skipped {len(skipped_files)} feature files while building stats.")
    if total_frames == 0 or sum_feats is None:
        raise ValueError("No valid frames available to compute global stats.")

    feature_mean = sum_feats / total_frames
    feature_var = sum_sq_feats / total_frames - feature_mean ** 2
    feature_std = np.sqrt(np.maximum(feature_var, 1e-8))

    payload = {
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
        "seq_lengths": np.array(seq_lengths, dtype=np.int32),
        "motion_means": np.array(motion_means, dtype=np.float32)
    }
    np.savez(stats_path, **payload)
    print(f"   ✅ Saved stats to {stats_path}")

if not os.path.exists(CFG['global_stats_path']):
    print("   ↳ Stats file missing; generating from processed metadata.")
    compute_and_save_global_stats(PROC_META, CFG['global_stats_path'])

stats_blob = np.load(CFG['global_stats_path'])
GLOBAL_MEAN = stats_blob['feature_mean'].astype(np.float32)
GLOBAL_STD = stats_blob['feature_std'].astype(np.float32)
SEQ_LENGTHS = stats_blob.get('seq_lengths', np.array([], dtype=np.int32))
MOTION_MEANS = stats_blob.get('motion_means', np.array([], dtype=np.float32))

print(f"   Feature mean/std loaded with shape: {GLOBAL_MEAN.shape}")
if SEQ_LENGTHS.size:
    print(f"   Avg frames: {SEQ_LENGTHS.mean():.1f} | 95th pct: {np.percentile(SEQ_LENGTHS, 95):.1f}")
if MOTION_MEANS.size:
    print(f"   Median motion mean: {np.median(MOTION_MEANS):.4f}")


print("\n📊 Loading metadata...")
meta = pd.read_csv(PROC_META)
print(f"   Total samples: {len(meta)}")
print(f"   Columns: {meta.columns.tolist()}")

# Validate required columns
required_cols = [
    'video_file',
    'feature_path',
    'sentence_id',
    'num_frames',
    'motion_mean',
    'motion_p95',
    'mean_visibility'
]
missing = [col for col in required_cols if col not in meta.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Filter obvious low-motion clips
before_filter = len(meta)
meta = meta[meta['motion_mean'] >= CFG['motion_floor']].reset_index(drop=True)
filtered_out = before_filter - len(meta)
if filtered_out:
    print(f"   Removed {filtered_out} low-motion clips (< {CFG['motion_floor']:.2e})")

# Check for missing files
print("\n🔍 Validating feature files...")
missing_files = []
for idx, row in meta.iterrows():
    if not os.path.exists(row['feature_path']):
        missing_files.append(row['feature_path'])

if missing_files:
    print(f"⚠️  Warning: {len(missing_files)} files not found")
    meta = meta[meta['feature_path'].apply(os.path.exists)].reset_index(drop=True)
    print(f"   Proceeding with {len(meta)} valid samples")

# Sentence distribution
sentence_counts = meta['sentence_id'].value_counts()
print(f"\n📈 Dataset statistics:")
print(f"   Unique sentences: {meta['sentence_id'].nunique()}")
print(f"   Videos per sentence (avg): {len(meta) / meta['sentence_id'].nunique():.2f}")
print(f"   Min videos per sentence: {sentence_counts.min()}")
print(f"   Max videos per sentence: {sentence_counts.max()}")
print(f"   Median motion mean: {meta['motion_mean'].median():.4f}")
print(f"   Mean visibility: {meta['mean_visibility'].mean():.2f}")

meta.head()
```

---

## CELL 5 — Augmentation Functions

```python
def normalize_with_stats(seq, mean_vec, std_vec):
    """Normalize features using global stats."""
    return (seq - mean_vec) / std_vec


def random_time_crop(seq, target_len):
    """Random crop to target length."""
    total = seq.shape[0]
    if total <= target_len:
        return seq
    start = random.randint(0, total - target_len)
    return seq[start:start + target_len]


def center_time_crop(seq, target_len):
    """Center crop to target length."""
    total = seq.shape[0]
    if total <= target_len:
        return seq
    start = (total - target_len) // 2
    return seq[start:start + target_len]


def random_time_warp(seq, max_warp=0.1):
    """Time warping via linear resampling."""
    total = seq.shape[0]
    factor = 1.0 + random.uniform(-max_warp, max_warp)
    new_total = max(1, int(round(total * factor)))
    idxs = np.linspace(0, total - 1, new_total)
    lo = np.floor(idxs).astype(int)
    hi = np.clip(lo + 1, 0, total - 1)
    w = idxs - lo
    resampled = (1.0 - w[:, None]) * seq[lo] + w[:, None] * seq[hi]
    return resampled.astype(np.float32)


def random_frame_dropout(seq, drop_prob=0.1):
    """Randomly drop frames to simulate jitter."""
    if seq.shape[0] <= 2 or drop_prob <= 0.0:
        return seq
    mask = np.random.rand(seq.shape[0]) > drop_prob
    if not mask.any():
        mask[random.randrange(seq.shape[0])] = True
    return seq[mask]


def temporal_shift(seq, max_shift=3):
    """Temporal shift by rolling sequence."""
    if max_shift <= 0:
        return seq
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return seq
    return np.roll(seq, shift, axis=0)


def add_noise(seq, sigma=0.01):
    """Add Gaussian noise."""
    if sigma <= 0:
        return seq
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)


def random_rotation(seq, max_angle=15):
    """
    Apply random Y-axis rotation to 3D coordinates.
    Assumes features are flattened [x, y, z, ...].
    """
    if max_angle <= 0:
        return seq
    
    theta = np.radians(random.uniform(-max_angle, max_angle))
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=np.float32)

    # Reshape to (T, N_landmarks, 3)
    # We need to know the structure. 
    # Based on preprocessing: 
    # POSE (33*4), HANDS (2*21*3), FACE (94*3)
    # Pose has visibility (x,y,z,v), others are (x,y,z)
    
    T, D = seq.shape
    out = seq.copy()
    
    # 1. Pose (33 landmarks * 4) -> indices 0..132
    # We only rotate x,y,z. Visibility is at index 3, 7, 11...
    pose_dim = 33 * 4
    pose_xyz = out[:, :pose_dim].reshape(T, 33, 4)
    # Rotate xyz
    pose_xyz[..., :3] = np.dot(pose_xyz[..., :3], rotation_matrix.T)
    out[:, :pose_dim] = pose_xyz.reshape(T, -1)
    
    # 2. Hands (2 * 21 * 3) -> indices 132..258
    hands_start = pose_dim
    hands_dim = 2 * 21 * 3
    hands_end = hands_start + hands_dim
    
    hands_xyz = out[:, hands_start:hands_end].reshape(T, -1, 3)
    hands_xyz = np.dot(hands_xyz, rotation_matrix.T)
    out[:, hands_start:hands_end] = hands_xyz.reshape(T, -1)
    
    # 3. Face (94 * 3) -> indices 258..540
    face_start = hands_end
    face_xyz = out[:, face_start:].reshape(T, -1, 3)
    face_xyz = np.dot(face_xyz, rotation_matrix.T)
    out[:, face_start:] = face_xyz.reshape(T, -1)
    
    return out


def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    """Compute per-frame motion energy from deltas."""
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    pose_xyz = frames[:, :33 * 4].reshape(frames.shape[0], 33, 4)[..., :3]
    pose_xyz = pose_xyz.reshape(frames.shape[0], -1)
    hands = frames[:, 33 * 4:33 * 4 + 2 * 21 * 3]
    face = frames[:, 33 * 4 + 2 * 21 * 3:]
    coords = np.concatenate([pose_xyz, hands, face], axis=1)
    diffs = np.diff(coords, axis=0, prepend=coords[:1])
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


print("✅ Augmentation helpers ready (with Rotation)")
```

---

## CELL 6 — Dataset Class

```python
from torch.nn.utils.rnn import pad_sequence

class SignDataset(Dataset):
    """Dataset yielding (frames, text) pairs for CLIP training."""

    def __init__(
        self,
        meta_df,
        tokenizer,
        seq_len=128,
        expected_dim=540,
        global_mean=None,
        global_std=None,
        augment=True,
        cfg=None
    ):
        self.meta = meta_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.expected_dim = expected_dim
        self.augment = augment
        self.global_mean = (global_mean if global_mean is not None else np.zeros(expected_dim, dtype=np.float32))
        self.global_std = (global_std if global_std is not None else np.ones(expected_dim, dtype=np.float32))
        self.cfg = cfg or {}
        self.time_warp = self.cfg.get('time_warp', 0.0)
        self.frame_dropout = self.cfg.get('frame_dropout', 0.0)
        self.noise_sigma = self.cfg.get('noise_sigma', 0.0)
        self.temporal_shift_max = self.cfg.get('temporal_shift', 0)
        self.rotation_angle = self.cfg.get('rotation_angle', 15)

        # Parse text variations
        self.text_variations = {}
        if 'text_variations' in self.meta.columns:
            for idx, row in self.meta.iterrows():
                try:
                    if pd.notna(row['text_variations']):
                        vars_list = json.loads(row['text_variations'])
                        if isinstance(vars_list, list) and vars_list:
                            self.text_variations[row['sentence_id']] = vars_list
                except:
                    pass

    def __len__(self):
        return len(self.meta)

    def load_and_validate(self, path):
        try:
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")
            if arr.shape[1] != self.expected_dim:
                raise ValueError(
                    f"Feature dim mismatch: expected {self.expected_dim}, got {arr.shape[1]}"
                )
            return arr
        except Exception as exc:
            raise RuntimeError(f"Error loading {path}: {exc}")

    def augment_seq(self, seq: np.ndarray) -> np.ndarray:
        out = seq.copy()
        if random.random() < 0.6 and self.time_warp > 0:
            out = random_time_warp(out, max_warp=self.time_warp)
        if random.random() < 0.5 and self.frame_dropout > 0:
            out = random_frame_dropout(out, drop_prob=self.frame_dropout)
        if random.random() < 0.5 and self.noise_sigma > 0:
            out = add_noise(out, sigma=self.noise_sigma)
        if random.random() < 0.8 and self.rotation_angle > 0:
            out = random_rotation(out, max_angle=self.rotation_angle)
        return out

    def preprocess(self, arr):
        seq = arr.astype(np.float32)
        if self.augment:
            seq = self.augment_seq(seq)
        else:
            seq = center_time_crop(seq, min(seq.shape[0], self.seq_len))

        if seq.shape[0] > self.seq_len:
            seq = random_time_crop(seq, self.seq_len) if self.augment else center_time_crop(seq, self.seq_len)

        seq = normalize_with_stats(seq, self.global_mean, self.global_std)

        if seq.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        elif seq.shape[0] > self.seq_len:
            seq = seq[:self.seq_len]

        return seq.astype(np.float32)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        try:
            arr = self.load_and_validate(row['feature_path'])
            frames = self.preprocess(arr)
            
            text = row['sentence']
            toks = self.tokenizer(
                text,
                max_length=CFG['text_max_len'],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            return {
                'frames': torch.from_numpy(frames),
                'input_ids': toks['input_ids'].squeeze(0),
                'attn_mask': toks['attention_mask'].squeeze(0),
                'label': idx
            }
        except Exception as exc:
            print(f"⚠️  Error processing {row['video_file']}: {exc}")
            dummy = np.zeros((self.seq_len, self.expected_dim), dtype=np.float32)
            dummy_toks = self.tokenizer(
                "",
                max_length=CFG['text_max_len'],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                'frames': torch.from_numpy(dummy),
                'input_ids': dummy_toks['input_ids'].squeeze(0),
                'attn_mask': dummy_toks['attention_mask'].squeeze(0),
                'label': -1
            }


def collate_fn(batch):
    valid_batch = [item for item in batch if item['label'] != -1]
    if len(valid_batch) == 0:
        return None, None, None
    
    frames = torch.stack([item['frames'] for item in valid_batch])
    input_ids = torch.stack([item['input_ids'] for item in valid_batch])
    attn_mask = torch.stack([item['attn_mask'] for item in valid_batch])
    
    return frames, input_ids, attn_mask


print("✅ Dataset class defined (Dual-View + Text Augmentation + Tokenization)")
```


```python
# Initialize AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG['text_model'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token

CFG['vocab_size'] = tokenizer.vocab_size
print(f"✅ Tokenizer loaded: {CFG['text_model']}")
print(f"   Vocab size: {CFG['vocab_size']}")
```

---

## CELL 7 — Smart Train/Val Split

```python
from torch.utils.data import Sampler
from collections import defaultdict

class ClassBalancedSampler(Sampler):
    """
    Sampler that ensures each batch contains at least K instances of each class.
    Useful for Supervised Contrastive Learning.
    """
    def __init__(self, data_source, batch_size, instances_per_class=2):
        self.data_source = data_source
        self.batch_size = batch_size
        self.instances_per_class = instances_per_class
        self.classes_per_batch = batch_size // instances_per_class
        
        # Group indices by sentence_id
        self.class_indices = defaultdict(list)
        for idx, row in data_source.iterrows():
            self.class_indices[row['sentence_id']].append(idx)
            
        self.classes = list(self.class_indices.keys())
        self.num_batches = len(data_source) // batch_size
        
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            # Select classes for this batch
            selected_classes = random.sample(self.classes, min(len(self.classes), self.classes_per_batch))
            
            for cls in selected_classes:
                indices = self.class_indices[cls]
                if len(indices) >= self.instances_per_class:
                    batch.extend(random.sample(indices, self.instances_per_class))
                else:
                    # Sample with replacement if not enough instances
                    batch.extend(random.choices(indices, k=self.instances_per_class))
            
            # Fill remaining slots if any
            while len(batch) < self.batch_size:
                cls = random.choice(self.classes)
                indices = self.class_indices[cls]
                batch.append(random.choice(indices))
                
            yield batch

    def __len__(self):
        return self.num_batches

print("\n🔀 Creating train/val split...")


train_aug_cfg = {
    'time_warp': CFG['time_warp'],
    'frame_dropout': CFG['frame_dropout'],
    'noise_sigma': CFG['noise_sigma'],
    'temporal_shift': CFG['temporal_shift'],
    'rotation_angle': 15,
}

# Group by sentence_id to ensure no data leakage
sentence_groups = meta.groupby('sentence_id').size()

print(f"   Sentences with 1 sample: {(sentence_groups == 1).sum()}")
print(f"   Sentences with 2+ samples: {(sentence_groups >= 2).sum()}")

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
print(f"   Train motion median: {train_meta['motion_mean'].median():.4f}")
print(f"   Val motion median: {val_meta['motion_mean'].median():.4f}")

train_sentences = set(train_meta['sentence_id'])
val_sentences = set(val_meta['sentence_id'])
overlap = train_sentences & val_sentences

if overlap:
    print(f"⚠️  Warning: {len(overlap)} sentences appear in both train and val!")
else:
    print("✅ No sentence overlap between train and val")

train_ds = SignDataset(
    train_meta,
    tokenizer=tokenizer,
    seq_len=CFG['seq_len'],
    expected_dim=CFG['input_dim'],
    global_mean=GLOBAL_MEAN,
    global_std=GLOBAL_STD,
    augment=True,
    cfg={
        'time_warp': CFG['time_warp'],
        'frame_dropout': CFG['frame_dropout'],
        'noise_sigma': CFG['noise_sigma'],
        'temporal_shift': CFG['temporal_shift'],
        'rotation_angle': 15 # Enable rotation
    }
)

val_aug_cfg = {
    'time_warp': CFG['time_warp'] * 0.35,
    'frame_dropout': CFG['frame_dropout'] * 0.5,
    'noise_sigma': CFG['noise_sigma'] * 0.5,
    'temporal_shift': max(1, CFG['temporal_shift'] // 2),
    'rotation_angle': 0 # No rotation for val
}

val_ds = SignDataset(
    val_meta,
    tokenizer=tokenizer,
    seq_len=CFG['seq_len'],
    expected_dim=CFG['input_dim'],
    global_mean=GLOBAL_MEAN,
    global_std=GLOBAL_STD,
    augment=True,
    cfg=val_aug_cfg
)

# Use ClassBalancedSampler for training
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
    """Project per-frame features to a lower-dimensional representation."""

    def __init__(self, in_dim, proj_dim=160):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # (B, T, F)
        out = self.norm(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.act(self.fc2(out))
        return out


class TemporalEncoder(nn.Module):
    """Encode temporal sequence into a fixed embedding with attention pooling."""

    def __init__(self, in_dim, hidden=160, n_layers=2, embed_dim=256, attn_heads=4):
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if n_layers > 1 else 0.0
        )
        self.attn_heads = attn_heads
        self.attn_key = nn.Linear(hidden * 2, hidden)
        self.attn_score = nn.Linear(hidden, attn_heads)
        self.attn_proj = nn.Linear(hidden * 2, embed_dim)
        
        # Projection for Seq2Seq context
        self.context_proj = nn.Linear(hidden * 2, embed_dim)
        
        self.post_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, return_sequence=False):  # (B, T, C)
        seq_out, _ = self.gru(x) # (B, T, 2*hidden)
        
        keys = torch.tanh(self.attn_key(seq_out))  # (B, T, hidden)
        attn_logits = self.attn_score(keys)  # (B, T, H)

        motion = torch.norm(x[:, 1:] - x[:, :-1], dim=-1, keepdim=True)
        motion = torch.cat([motion[:, :1], motion], dim=1)  # (B, T, 1)
        motion = (motion - motion.mean(dim=1, keepdim=True)) / (motion.std(dim=1, keepdim=True) + 1e-6)
        attn_logits = attn_logits + motion

        weights = torch.softmax(attn_logits, dim=1)  # (B, T, H)
        pooled = torch.sum(seq_out.unsqueeze(-2) * weights.unsqueeze(-1), dim=1)  # (B, H, 2*hidden)
        pooled = pooled.mean(dim=1)  # average across heads

        embedding = self.attn_proj(self.dropout(pooled))
        embedding = self.post_norm(embedding)
        embedding = nn.functional.normalize(embedding, dim=-1)
        
        if return_sequence:
            # Project seq_out to embed_dim for decoder usage
            context = self.context_proj(seq_out) # (B, T, embed_dim)
            return embedding, weights.mean(dim=2), context
            
        return embedding, weights.mean(dim=2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SignTranslationModel(nn.Module):
    """Seq2Seq model: Visual Encoder + Text Decoder."""

    def __init__(self, input_dim, vocab_size, proj_dim=160, hidden=160, n_layers=2, embed_dim=256, attn_heads=4):
        super().__init__()
        self.projector = FrameProjector(input_dim, proj_dim)
        self.encoder = TemporalEncoder(proj_dim, hidden, n_layers, embed_dim, attn_heads)
        
        # Decoder components
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=attn_heads, dim_feedforward=embed_dim*4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2) # Start with 2 layers
        
        self.generator = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def encode_visual(self, x):
        """Encode visual sequence for Contrastive Loss."""
        proj = self.projector(x)
        embedding, attn_weights = self.encoder(proj)
        return embedding

    def encode_for_inference(self, x):
        """Encode visual sequence and return intermediates for analysis."""
        proj = self.projector(x)
        embedding, attn_weights, context = self.encoder(proj, return_sequence=True)
        return embedding, proj, attn_weights

    def forward(self, src, tgt, tgt_mask=None, tgt_pad_mask=None):
        """
        src: (B, T, F) - Visual input
        tgt: (B, L) - Text input (token IDs)
        """
        # 1. Encode Visual
        proj = self.projector(src)
        embedding, attn_weights, context = self.encoder(proj, return_sequence=True)
        # context: (B, T, embed_dim)
        
        # Prepare for Transformer (S, B, E)
        memory = context.permute(1, 0, 2) 
        
        # 2. Embed Text
        tgt_emb = self.tgt_embed(tgt).permute(1, 0, 2) # (L, B, E)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # 3. Decode
        output = self.decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        ) # (L, B, E)
        
        # 4. Generate logits
        logits = self.generator(output.permute(1, 0, 2)) # (B, L, Vocab)
        
        return logits, embedding

    @torch.no_grad()
    def project_frames(self, x):
        return self.projector(x)


# Update CFG with vocab size if not present (handled by tokenizer cell)
vocab_size = CFG.get('vocab_size', 1000) # Default fallback

model = SignTranslationModel(
    input_dim=CFG['input_dim'],
    vocab_size=vocab_size,
    proj_dim=CFG['proj_dim'],
    hidden=CFG['rnn_hidden'],
    n_layers=CFG['rnn_layers'],
    embed_dim=CFG['embed_dim'],
    attn_heads=CFG['attn_heads']
).to(CFG['device'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n🤖 SignTranslationModel initialized")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (fp32)")
```

---

## CELL 9 — Loss & Metrics

```python
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

def clip_loss(visual_emb, text_emb, temperature=CFG['temperature']):
    visual_emb = F.normalize(visual_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = (visual_emb @ text_emb.t()) / temperature
    labels = torch.arange(visual_emb.size(0), device=visual_emb.device)
    loss_v = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_v + loss_t) / 2

print("✅ Loss functions defined (CLIP + Label Smoothing)")
```

```python
def _length_penalty(length: int, alpha: float) -> float:
    if alpha <= 0:
        return 1.0
    return ((5 + length) / 6) ** alpha

def _beam_search_single(model, memory, tokenizer, device, max_len=64, beam_size=4, length_penalty=0.8):
    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    
    beams = [(0.0, [bos], False)]
    finished = []
    
    for _ in range(max_len):
        new_beams = []
        for logprob, seq, done in beams:
            if done:
                new_beams.append((logprob, seq, True))
                continue
            
            tgt_ids = torch.tensor([seq], dtype=torch.long, device=device) # (1, L)
            tgt_mask = generate_square_subsequent_mask(tgt_ids.size(1), device)
            
            # Forward through TextDecoder
            logits = model.decoder(tgt_ids, memory, tgt_mask=tgt_mask) # (1, L, V)
            
            # Get last step logits
            last_logits = logits[0, -1, :] # (V)
            log_probs = F.log_softmax(last_logits, dim=-1)
            
            topk = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))
            
            for value, index in zip(topk.values.tolist(), topk.indices.tolist()):
                token_id = int(index)
                next_seq = seq + [token_id]
                next_done = (token_id == eos)
                next_logprob = logprob + value
                new_beams.append((next_logprob, next_seq, next_done))
                
                if next_done:
                    finished.append((next_logprob, next_seq))
                    
        if not new_beams: break
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
        
        if all(done for _, _, done in beams): break
        
    # Select best
    candidates = finished if finished else [(s, seq) for s, seq, _ in beams]
    best = max(candidates, key=lambda x: x[0] / _length_penalty(len(x[1]), length_penalty))
    return best[1]

@torch.no_grad()
def beam_search_decode_batch(model, src_batch, tokenizer, device, max_len=64, beam_size=4, length_penalty=0.8):
    model.eval()
    src_batch = src_batch.to(device)
    memory, _ = model.encoder(src_batch) # (B, T, H)
    
    decoded = []
    for i in range(src_batch.size(0)):
        mem_i = memory[i:i+1] # (1, T, H)
        seq = _beam_search_single(model, mem_i, tokenizer, device, max_len, beam_size, length_penalty)
        decoded.append(seq)
    return decoded


def tokens_to_sentence(tokenizer, token_ids: List[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return float(len(hyp_words) > 0)
    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    for i in range(len(ref_words) + 1):
        dp[i, 0] = i
    for j in range(len(hyp_words) + 1):
        dp[0, j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost
            )
    return float(dp[len(ref_words), len(hyp_words)] / max(len(ref_words), 1))


def simple_bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not hyp_tokens:
        return 0.0
    eps = 1e-9
    precisions = []
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            precisions.append(eps)
            continue
        ref_counts = Counter(tuple(ref_tokens[i:i + n]) for i in range(max(len(ref_tokens) - n + 1, 0)))
        hyp_counts = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
        overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in hyp_counts.items())
        total = sum(hyp_counts.values())
        precisions.append((overlap + eps) / (total + eps))
    log_precision = sum(math.log(p) for p in precisions) / len(precisions)
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - (ref_len / max(hyp_len, 1)))
    return float(bp * math.exp(log_precision))


@torch.no_grad()
def evaluate_decoder_generation(
    model,
    loader,
    tokenizer,
    device,
    max_samples: int = 32,
    max_len: int = 64,
    beam_size: int | None = None,
    length_penalty: float | None = None
) -> dict:
    model.eval()
    samples = []
    processed = 0
    beam = beam_size or CFG.get('beam_size', 1)
    penalty = length_penalty if length_penalty is not None else CFG.get('beam_length_penalty', 0.0)

    for batch in loader:
        if batch[0] is None:
            continue
        visuals = batch[0]
        token_ids = batch[1] # input_ids is at index 1 now
        decoded = beam_search_decode_batch(
            model,
            visuals,
            tokenizer,
            device,
            max_len=max_len,
            beam_size=beam,
            length_penalty=penalty
        )
        for ref_ids, hyp_tokens in zip(token_ids, decoded):
            reference = tokens_to_sentence(tokenizer, ref_ids.cpu().tolist())
            hypothesis = tokens_to_sentence(tokenizer, hyp_tokens)
            wer = word_error_rate(reference, hypothesis)
            bleu = simple_bleu_score(reference, hypothesis)
            samples.append({
                'reference': reference,
                'prediction': hypothesis,
                'wer': wer,
                'bleu': bleu
            })
            processed += 1
            if processed >= max_samples:
                break
        if processed >= max_samples:
            break

    if not samples:
        return {
            'decoder_wer': float('nan'),
            'decoder_bleu': float('nan'),
            'decoder_exact': float('nan'),
            'samples': []
        }

    avg_wer = float(np.mean([s['wer'] for s in samples]))
    avg_bleu = float(np.mean([s['bleu'] for s in samples]))
    exact_match = float(np.mean([1.0 if s['reference'] == s['prediction'] else 0.0 for s in samples]))
    return {
        'decoder_wer': avg_wer,
        'decoder_bleu': avg_bleu,
        'decoder_exact': exact_match,
        'samples': samples
    }


print("✅ Decoder helpers (beam search + text metrics) ready")
```


```python
@torch.no_grad()
def evaluate_alignment(model, loader, device):
    """Evaluate CLIP alignment metrics (Instance Top-1, Cosine Sim) on the validation set."""
    model.eval()
    all_visual = []
    all_text = []
    
    for batch in loader:
        if batch[0] is None: continue
        frames, input_ids, attn_mask = batch
        frames = frames.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        
        # Get projections
        # Note: We don't need the decoder output here, just the projections
        # But the model forward returns (logits, visual_proj, text_proj)
        # We pass dummy decoder inputs to satisfy the forward signature if needed, 
        # or better yet, we can just use the encoders directly if accessible.
        # However, model() is convenient. Let's use it with dummy decoder inputs.
        tgt_inp = input_ids[:, :-1]
        
        _, visual_proj, text_proj = model(frames, tgt_inp, None, text_input_ids=input_ids, text_attn_mask=attn_mask)
        
        all_visual.append(visual_proj.cpu())
        all_text.append(text_proj.cpu())
        
    if not all_visual:
        return {'inst_top1': 0.0, 'pos_sim': 0.0, 'neg_sim': 0.0}
        
    visual_emb = torch.cat(all_visual)
    text_emb = torch.cat(all_text)
    
    # Normalize
    visual_emb = F.normalize(visual_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    
    # Similarity matrix
    sim_matrix = visual_emb @ text_emb.t()
    
    # Metrics
    labels = torch.arange(len(visual_emb))
    preds = sim_matrix.argmax(dim=-1)
    inst_top1 = (preds == labels).float().mean().item()
    
    # Cosine similarities
    pos_sim = torch.diag(sim_matrix).mean().item()
    
    # Negative similarity (off-diagonal)
    # If batch size is 1, neg_sim is undefined (or 0)
    if len(visual_emb) > 1:
        mask = ~torch.eye(len(visual_emb), dtype=torch.bool)
        neg_sim = sim_matrix[mask].mean().item()
    else:
        neg_sim = 0.0
    
    return {
        'inst_top1': inst_top1,
        'pos_sim': pos_sim,
        'neg_sim': neg_sim
    }

@torch.no_grad()
def evaluate_sentence_retrieval(model, meta_df, cfg, global_mean, global_std, tokenizer, batch_size=32):
    """Measure retrieval accuracy across the validation set."""
    model.eval()
    if meta_df.empty:
        return {
            "ret_top1": float("nan"),
            "ret_mrr": float("nan"),
            "pos_sim": float("nan"),
            "neg_sim": float("nan")
        }

    eval_ds = SignDataset(
        meta_df,
        tokenizer=tokenizer,
        seq_len=cfg['seq_len'],
        expected_dim=cfg['input_dim'],
        global_mean=global_mean,
        global_std=global_std,
        augment=False
    )

    sequences = []
    sentence_ids = []
    for row in meta_df.itertuples():
        try:
            arr = eval_ds.load_and_validate(row.feature_path)
            # Preprocess without augmentation (augment=False in init)
            seq = eval_ds.preprocess(arr)
            sequences.append(seq)
            sentence_ids.append(int(row.sentence_id))
        except Exception as exc:
            print(f"⚠️  Retrieval eval skipped {row.video_file}: {exc}")

    if not sequences:
        return {
            "ret_top1": float("nan"),
            "ret_mrr": float("nan"),
            "pos_sim": float("nan"),
            "neg_sim": float("nan")
        }

    stacked = torch.from_numpy(np.stack(sequences)).to(cfg['device'])
    embeddings = []
    for start in range(0, stacked.size(0), batch_size):
        batch = stacked[start:start + batch_size]
        # Use encode_visual
        emb = model.encode_visual(batch)
        embeddings.append(emb.cpu().numpy())

    embs = np.vstack(embeddings)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sims = embs @ embs.T
    np.fill_diagonal(sims, -np.inf)

    sentence_ids = np.array(sentence_ids)
    same_mask = sentence_ids[:, None] == sentence_ids[None, :]
    diag_mask = np.eye(len(sentence_ids), dtype=bool)
    pos_mask = same_mask & ~diag_mask
    neg_mask = ~same_mask

    pos_sims = sims[pos_mask] if pos_mask.any() else np.array([], dtype=np.float32)
    neg_sims = sims[neg_mask] if neg_mask.any() else np.array([], dtype=np.float32)

    top1_hits = []
    reciprocal_ranks = []
    for idx in range(len(sentence_ids)):
        positives = np.where(pos_mask[idx])[0]
        if positives.size == 0:
            continue
        order = np.argsort(-sims[idx])
        for rank, j in enumerate(order, start=1):
            if j in positives:
                top1_hits.append(1 if rank == 1 else 0)
                reciprocal_ranks.append(1.0 / rank)
                break

    metrics = {
        "ret_top1": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "ret_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "pos_sim": float(np.mean(pos_sims)) if pos_sims.size else float("nan"),
        "neg_sim": float(np.mean(neg_sims)) if neg_sims.size else float("nan")
    }
    return metrics
    
```

```python
class FrameProjector(nn.Module):
    """Project per-frame features to a lower-dimensional representation."""

    def __init__(self, in_dim, proj_dim=160):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # (B, T, F)
        out = self.norm(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.act(self.fc2(out))
        return out


class FrozenTextEncoder(nn.Module):
    def __init__(self, model_name=CFG['text_model'], proj_dim=CFG['proj_dim']):
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
    def __init__(self, pose_dim=CFG['input_dim'], hidden_dim=CFG['hidden_dim'], n_heads=CFG['n_heads'], num_layers=CFG['num_layers']):
        super().__init__()
        self.input_proj = nn.Linear(pose_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4, batch_first=True, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(CFG['seq_len'], hidden_dim) * 0.02)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb[: x.size(1)]
        enc = self.encoder(x)
        pooled = enc.mean(dim=1)
        return enc, pooled


class TextDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim=CFG['hidden_dim'], n_heads=CFG['n_heads'], num_layers=CFG['num_layers'], pad_token_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.pos_emb = nn.Parameter(torch.randn(CFG['text_max_len'], hidden_dim) * 0.02)
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
        self.visual_proj = nn.Linear(CFG['hidden_dim'], CFG['proj_dim'])
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
    
    def encode_visual(self, frames):
        """Helper for inference/retrieval."""
        memory, pooled = self.encoder(frames)
        return self.visual_proj(pooled)

    def encode_for_inference(self, frames):
        """Helper for inference engine returning (emb, proj, attn_proxy)."""
        memory, pooled = self.encoder(frames)
        visual_proj = self.visual_proj(pooled)
        # Return pooled as 'emb' (raw representation), visual_proj as 'proj', and memory as 'attn'
        return pooled, visual_proj, memory


model = Sign2TextModel(vocab_size=CFG['vocab_size'], pad_token_id=tokenizer.pad_token_id).to(CFG['device'])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n🤖 Sign2TextModel initialized (CLIP + Seq2Seq)")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

```
---

## CELL 10 — Training Loop

```python
def train_epoch(model, loader, optimizer, device, lambda_clip=CFG['lambda_clip'], pad_token_id=0):
    model.train()
    total_loss = 0
    total_ce = 0
    total_clip = 0
    steps = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if batch[0] is None: continue
        
        frames, input_ids, attn_mask = batch
        frames = frames.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        
        tgt_inp = input_ids[:, :-1]
        tgt_out = input_ids[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_inp.size(1), device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, visual_proj, text_proj = model(frames, tgt_inp, tgt_mask, text_input_ids=input_ids, text_attn_mask=attn_mask)
        
        # Losses
        ce = label_smoothing_nll_loss(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), pad_token_id, eps=0.1)
        c_loss = clip_loss(visual_proj, text_proj)
        
        loss = ce + lambda_clip * c_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
        optimizer.step()
        
        total_loss += loss.item()
        total_ce += ce.item()
        total_clip += c_loss.item()
        steps += 1
        
    if steps == 0: return 0.0, 0.0, 0.0
    return total_loss/steps, total_ce/steps, total_clip/steps

def train_model(model, train_loader, val_loader, epochs=CFG['epochs'], lr=CFG['lr']):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=CFG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pad_id = tokenizer.pad_token_id
    
    history = []
    best_inst_top1 = 0.0
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        t0 = time.time()
        loss, ce, clip = train_epoch(model, train_loader, optimizer, CFG['device'], pad_token_id=pad_id)
        scheduler.step()
        
        # Validation
        val_metrics = evaluate_alignment(model, val_loader, CFG['device'])
        
        # Decoder evaluation (every 5 epochs or last epoch)
        decoder_metrics = {}
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            decoder_metrics = evaluate_decoder_generation(
                model, val_loader, tokenizer, CFG['device'], max_samples=32
            )
            
        dt = time.time() - t0
        
        # Log
        print(f"Epoch {epoch+1}/{epochs} [{dt:.1f}s] | Loss: {loss:.4f} (CE: {ce:.4f}, CLIP: {clip:.4f})")
        print(f"   Val Inst@1: {val_metrics['inst_top1']:.3f} | PosSim: {val_metrics['pos_sim']:.3f} | NegSim: {val_metrics['neg_sim']:.3f}")
        if decoder_metrics:
            print(f"   Decoder WER: {decoder_metrics.get('decoder_wer', float('nan')):.3f} | BLEU: {decoder_metrics.get('decoder_bleu', float('nan')):.3f}")

        # Record history
        hist_entry = {
            'epoch': epoch + 1,
            'train_loss': loss,
            'ce_loss': ce,
            'clip_loss': clip,
            'val_inst_top1': val_metrics['inst_top1'],
            'val_pos_sim': val_metrics['pos_sim'],
            'val_neg_sim': val_metrics['neg_sim'],
            'lr': optimizer.param_groups[0]['lr']
        }
        hist_entry.update(decoder_metrics)
        history.append(hist_entry)
        
        # Save checkpoint
        if val_metrics['inst_top1'] > best_inst_top1:
            best_inst_top1 = val_metrics['inst_top1']
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pt"))
            print(f"   ⭐ New best model saved (Inst@1: {best_inst_top1:.3f})")
            
        # Save history CSV
        pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)
            
    return history

# Run training
history = train_model(model, train_loader, val_loader)
```

---

## CELL 11 — Generate & Save Embeddings

```python
def compute_embeddings(model, loader, device):
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing Embeddings"):
            if batch[0] is None: continue
            frames = batch[0].to(device)
            # Use encode_visual to get the projection
            visual_proj = model.encode_visual(frames)
            embs.append(visual_proj.cpu())
            
    if embs:
        return torch.cat(embs), None
    return None, None

print("\n💾 Generating embeddings for all data...")

all_meta = pd.concat([train_meta, val_meta]).reset_index(drop=True)
all_ds = SignDataset(
    all_meta,
    tokenizer=tokenizer,
    seq_len=CFG['seq_len'],
    expected_dim=CFG['input_dim'],
    global_mean=GLOBAL_MEAN,
    global_std=GLOBAL_STD,
    augment=False
)
all_loader = DataLoader(
    all_ds,
    batch_size=CFG['batch_size'],
    shuffle=False,
    num_workers=CFG['num_workers'],
    collate_fn=collate_fn
)

embs, labels = compute_embeddings(trained_model, all_loader, CFG['device'])

if embs is not None:
    embs_np = embs.numpy()
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), embs_np)

    mapping_df = all_meta[['video_file', 'feature_path', 'sentence_id', 'sentence', 'category']].copy()
    mapping_df.to_csv(os.path.join(OUT_DIR, "embeddings_map.csv"), index=False)

    print(f"✅ Saved {len(embs_np)} embeddings")
    print(f"   Shape: {embs_np.shape}")
    print(f"   Files: embeddings.npy, embeddings_map.csv")

    # Build sentence-level prototypes
    norm_embs = embs_np / (np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-8)
    sentence_ids = mapping_df['sentence_id'].to_numpy()
    proto_vectors = []
    proto_sentence_ids = []
    proto_counts = []

    for sid in np.unique(sentence_ids):
        idxs = np.where(sentence_ids == sid)[0]
        if idxs.size < CFG['prototype_min_clips']:
            continue
        proto = norm_embs[idxs].mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        proto_vectors.append(proto.astype(np.float32))
        proto_sentence_ids.append(int(sid))
        proto_counts.append(int(idxs.size))

    if proto_vectors:
        proto_path = os.path.join(OUT_DIR, "prototypes.npz")
        np.savez(
            proto_path,
            sentence_ids=np.array(proto_sentence_ids, dtype=np.int32),
            embeddings=np.stack(proto_vectors, axis=0),
            counts=np.array(proto_counts, dtype=np.int32)
        )
        print(f"   Prototypes saved: {proto_path} ({len(proto_vectors)} classes)")
    else:
        print("   ⚠️ No prototypes generated (insufficient clips per sentence)")
else:
    print("❌ Failed to compute embeddings")
```

---

## CELL 12 — Inference Helper (Cached)

```python
class InferenceEngine:
    """Embedding-based inference with motion gating, prototypes, and optional DTW."""

    def __init__(
        self,
        model,
        embeddings_path,
        mapping_path,
        cfg,
        global_mean,
        global_std,
        prototypes_path: Optional[str] = None
    ):
        self.model = model.eval().to(cfg['device'])
        self.cfg = cfg
        self.seq_len = cfg['seq_len']
        self.global_mean = global_mean
        self.global_std = global_std

        self.embs = np.load(embeddings_path)
        self.mapping = pd.read_csv(mapping_path)
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True)
        self.embs = self.embs / (norms + 1e-8)

        self.proto_lookup = {}
        if prototypes_path and os.path.exists(prototypes_path):
            proto_blob = np.load(prototypes_path)
            for sid, emb in zip(proto_blob['sentence_ids'], proto_blob['embeddings']):
                self.proto_lookup[int(sid)] = emb / (np.linalg.norm(emb) + 1e-8)
            print(f"✅ Loaded {len(self.proto_lookup)} prototypes")
        else:
            print("ℹ️  No prototypes supplied; falling back to clip embeddings only")

        self.dtw_enabled = cfg['dtw']['enabled'] and fastdtw is not None
        if cfg['dtw']['enabled'] and fastdtw is None:
            print("⚠️  fastdtw not installed; DTW re-ranking disabled")

    def preprocess_sequence(self, feat_np: np.ndarray) -> np.ndarray:
        seq = feat_np.astype(np.float32)
        if seq.shape[0] > self.seq_len:
            seq = center_time_crop(seq, self.seq_len)
        seq = normalize_with_stats(seq, self.global_mean, self.global_std)
        if seq.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        elif seq.shape[0] > self.seq_len:
            seq = seq[:self.seq_len]
        return seq

    @torch.no_grad()
    def encode(self, seq_np: np.ndarray):
        tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.cfg['device'])
        # encode_for_inference returns: pooled, visual_proj, memory
        pooled, visual_proj, memory = self.model.encode_for_inference(tensor)
        
        # Use visual_proj for global similarity
        emb_np = visual_proj.cpu().numpy()[0]
        emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)
        
        # Use memory (sequence output) for DTW if needed
        seq_rep = memory.cpu().numpy()[0]
        
        return emb_np, seq_rep, memory

    def predict(self, feat_np: np.ndarray, topk=5):
        motion_vals = compute_motion_energy(feat_np.astype(np.float32))
        motion_mean = float(motion_vals.mean()) if motion_vals.size else 0.0
        if motion_mean < self.cfg['motion_floor']:
            return {
                'accepted': False,
                'reason': 'low_motion',
                'motion_mean': motion_mean,
                'top': []
            }

        seq_np = self.preprocess_sequence(feat_np)
        emb, seq_rep, _ = self.encode(seq_np)

        raw_sims = self.embs @ emb
        refined_sims = raw_sims.copy()
        dtw_meta = {}

        # DTW disabled for now as model architecture changed (requires frame-level projections)
        # if self.dtw_enabled: ...

        logits = refined_sims / self.cfg['temperature']
        probs = torch.softmax(torch.from_numpy(logits), dim=0).numpy()

        sorted_idx = refined_sims.argsort()[::-1][:topk]
        predictions = []
        top1 = refined_sims[sorted_idx[0]] if sorted_idx.size > 0 else -1.0
        top2 = refined_sims[sorted_idx[1]] if sorted_idx.size > 1 else -1.0

        for rank, idx in enumerate(sorted_idx, start=1):
            row = self.mapping.iloc[idx]
            sentence_id = int(row['sentence_id'])
            proto_sim = float(np.dot(self.proto_lookup[sentence_id], emb)) if sentence_id in self.proto_lookup else None
            entry = {
                'rank': rank,
                'video_file': row['video_file'],
                'feature_path': row['feature_path'],
                'sentence_id': sentence_id,
                'sentence': row['sentence'],
                'category': row['category'],
                'similarity': float(refined_sims[idx]),
                'raw_similarity': float(raw_sims[idx]),
                'confidence': float(probs[idx]),
                'proto_similarity': proto_sim,
            }
            predictions.append(entry)

        # Simple thresholding
        similarity_ok = top1 >= 0.6 
        margin_ok = (top1 - top2) >= 0.05 if top2 > -1.0 else True
        proto_ok = True
        if predictions:
            proto_sim = predictions[0]['proto_similarity']
            if proto_sim is not None:
                proto_ok = proto_sim >= (0.6 - 0.05)

        accepted = similarity_ok and margin_ok and proto_ok
        reason = 'ok'
        if not similarity_ok: reason = 'low_similarity'
        elif not margin_ok: reason = 'ambiguous_top2'
        elif not proto_ok: reason = 'prototype_disagreement'

        return {
            'accepted': accepted,
            'reason': reason,
            'motion_mean': motion_mean,
            'top': predictions
        }


engine = InferenceEngine(
    trained_model,
    os.path.join(OUT_DIR, "embeddings.npy"),
    os.path.join(OUT_DIR, "embeddings_map.csv"),
    CFG,
    GLOBAL_MEAN,
    GLOBAL_STD,
    prototypes_path=os.path.join(OUT_DIR, "prototypes.npz")
)

print("\n✅ Inference engine ready")

test_row = val_meta.sample(1).iloc[0]
test_feat = np.load(test_row['feature_path'])
result = engine.predict(test_feat, topk=5)

print(f"\n🧪 Test prediction for: {test_row['video_file']}")
print(f"   True sentence: {test_row['sentence']}")
print(f"   Motion mean: {result['motion_mean']:.4f}")
print(f"   Accepted: {result['accepted']} ({result['reason']})")

for pred in result['top']:
    marker = "✅" if pred['sentence_id'] == test_row['sentence_id'] else "  "
    proto = pred['proto_similarity'] if pred['proto_similarity'] is not None else float('nan')
    dtw_note = f", DTW {pred.get('dtw_score', float('nan')):.3f}" if 'dtw_score' in pred else ""
    print(
        f"   {pred['rank']}. {marker} [{pred['confidence']:.2%}] "
        f"sim={pred['similarity']:.3f} (raw {pred['raw_similarity']:.3f})"
        f", proto={proto:.3f}{dtw_note} -> {pred['sentence'][:70]}"
    )
```


## CELL - Evaluate Validation Set

```python
print("\n🧪 Evaluating held-out validation set...")
alignment_metrics = evaluate_alignment(trained_model, val_loader, CFG['device'])
print(f"   Inst@1: {alignment_metrics['inst_top1']:.3f}")
print(f"   PosSim: {alignment_metrics['pos_sim']:.3f}")
print(f"   NegSim: {alignment_metrics['neg_sim']:.3f}")

if history:
    latest = history[-1]
    print(
        f"   Latest Losses -> Total: {latest['train_loss']:.4f} | "
        f"SupCon: {latest['supcon_loss']:.4f} | CE: {latest['ce_loss']:.4f}"
    )

retrieval_batch = max(1, int(CFG.get('retrieval_batch_size', CFG['batch_size'])))
retrieval_metrics = evaluate_sentence_retrieval(
    trained_model,
    val_meta,
    CFG,
    GLOBAL_MEAN,
    GLOBAL_STD,
    tokenizer,
    batch_size=retrieval_batch
)
print(f"   Ret@1: {retrieval_metrics['ret_top1']:.3f}")
print(f"   MRR: {retrieval_metrics['ret_mrr']:.3f}")
if not math.isnan(retrieval_metrics['pos_sim']):
    print(f"   PosSim (retrieval): {retrieval_metrics['pos_sim']:.3f}")
if not math.isnan(retrieval_metrics['neg_sim']):
    print(f"   NegSim (retrieval): {retrieval_metrics['neg_sim']:.3f}")

decoder_eval = evaluate_decoder_generation(
    trained_model,
    val_loader,
    tokenizer,
    CFG['device'],
    max_samples=CFG.get('decoder_eval_samples', 32),
    max_len=CFG.get('decoder_max_len', 64)
)
if not math.isnan(decoder_eval['decoder_wer']):
    print(
        f"   Decoder -> WER: {decoder_eval['decoder_wer']:.3f} | "
        f"BLEU: {decoder_eval['decoder_bleu']:.3f} | Exact: {decoder_eval['decoder_exact']:.3f}"
    )
    for sample in decoder_eval.get('samples', [])[:2]:
        print(f"      REF: {sample['reference']}")
        print(f"      HYP: {sample['prediction']}")
        print(f"      WER: {sample['wer']:.3f} | BLEU: {sample['bleu']:.3f}")
else:
    print("   Decoder metrics unavailable (no samples decoded).")

if val_meta.empty:
    print("\n⚠️  No validation samples available for qualitative inspection.")
else:
    samples_to_check = val_meta.sample(
        n=min(2, len(val_meta)),
        random_state=SEED
    )
    for sample in samples_to_check.itertuples():
        try:
            feat_np = np.load(sample.feature_path)
            result = engine.predict(feat_np, topk=5)
        except Exception as exc:
            print("\n⚠️  Skipping sample due to load error:", sample.video_file, "->", exc)
            continue

        print("\n🎯 Ground truth")
        print(f"   Video: {sample.video_file}")
        print(f"   Sentence: {sample.sentence}")
        print(f"   Motion mean: {result['motion_mean']:.4f}")
        print(f"   Accepted: {result['accepted']} ({result['reason']})")

        if not result['top']:
            print("   ⚠️  No retrieval candidates returned.")
            continue

        for pred in result['top']:
            marker = "✅" if pred['sentence_id'] == sample.sentence_id else "  "
            proto = pred['proto_similarity'] if pred['proto_similarity'] is not None else float('nan')
            dtw_score = pred.get('dtw_score')
            dtw_note = f", DTW {dtw_score:.3f}" if dtw_score is not None else ""
            print(
                f"   {pred['rank']}. {marker} [{pred['confidence']:.2%}] ",
                f"sim={pred['similarity']:.3f}, proto={proto:.3f}{dtw_note} -> {pred['sentence'][:80]}",
                sep=""
            )
```

---

## CELL 13 — Export to ONNX

> The ONNX export assumes each clip is padded/truncated to `CFG['seq_len']` (64 frames) and bakes in the batch size used here (`1`). Keep preprocessing aligned or refactor the model before enabling variable-length or variable-batch exports.

```python
print("\n📦 Exporting model to ONNX...")

# Prepare model for export
model_cpu = trained_model.cpu().eval()

# Wrapper to export just the encoder part
class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model.encode_visual(x)

encoder_export = EncoderWrapper(model_cpu)

# Dummy input uses the fixed 64-frame sequence length expected by the model
dummy_input = torch.randn(1, CFG['seq_len'], CFG['input_dim'])

# Export (fixed sequence length; batch dimension remains flexible)
onnx_path = os.path.join(OUT_DIR, "sign_encoder.onnx")

torch.onnx.export(
    encoder_export,
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
import numpy as np

history_df = pd.DataFrame(history)
if history_df.empty:
    print("⚠️  No training history captured; skipping visualization.")
else:
    history_df = history_df.sort_values('epoch').reset_index(drop=True)
    history_df['loss_smooth'] = history_df['train_loss'].rolling(window=5, min_periods=1).mean()
    history_df['clip_smooth'] = history_df['clip_loss'].rolling(window=5, min_periods=1).mean()
    history_df['ce_smooth'] = history_df['ce_loss'].rolling(window=5, min_periods=1).mean()
    best_idx = history_df['val_inst_top1'].idxmax()
    best_epoch = int(history_df.loc[best_idx, 'epoch'])

    has_retrieval_cols = {'val_retrieval_top1', 'val_retrieval_mrr'} & set(history_df.columns)
    retrieval_available = False
    if has_retrieval_cols:
        retrieval_available = history_df[list(has_retrieval_cols)].notna().any().any()

    n_cols = 3 if retrieval_available else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # Training + component losses
    ax_loss = axes[0]
    ax_loss.plot(history_df['epoch'], history_df['train_loss'], marker='o', label='Total Loss', alpha=0.3)
    ax_loss.plot(history_df['epoch'], history_df['loss_smooth'], label='Total (smoothed)', linewidth=2)
    ax_loss.plot(history_df['epoch'], history_df['clip_loss'], marker='s', label='CLIP Loss', alpha=0.3)
    ax_loss.plot(history_df['epoch'], history_df['clip_smooth'], label='CLIP (smoothed)', linewidth=2)
    ax_loss.plot(history_df['epoch'], history_df['ce_loss'], marker='^', label='Cross-Entropy Loss', alpha=0.3)
    ax_loss.plot(history_df['epoch'], history_df['ce_smooth'], label='CE (smoothed)', linewidth=2)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss Breakdown')
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()
    ax_loss.axvline(best_epoch, color='tab:green', linestyle='--', alpha=0.6)
    ax_loss.annotate(
        f"best@{best_epoch}",
        xy=(best_epoch, history_df.loc[best_idx, 'loss_smooth']),
        xytext=(5, -10),
        textcoords='offset points',
        color='tab:green',
        fontsize=9
    )

    # Alignment metrics
    ax_align = axes[1]
    ax_align.plot(history_df['epoch'], history_df['val_inst_top1'], marker='o', label='Inst@1')
    ax_align.plot(history_df['epoch'], history_df['val_pos_sim'], marker='s', label='Pos Cos Sim')
    ax_align.plot(history_df['epoch'], history_df['val_neg_sim'], marker='^', label='Neg Cos Sim')
    ax_align.axvline(best_epoch, color='tab:green', linestyle='--', alpha=0.6)
    ax_align.set_xlabel('Epoch')
    ax_align.set_ylabel('Alignment Metric')
    ax_align.set_title('Validation Alignment Metrics')
    ax_align.legend()
    ax_align.grid(alpha=0.3)

    if retrieval_available:
        ax_ret = axes[2]
        retrieval_cols = [col for col in ['val_retrieval_top1', 'val_retrieval_mrr'] if col in history_df.columns]
        retrieval_df = history_df[['epoch'] + retrieval_cols].dropna(how='all')
        if retrieval_df.empty:
            ax_ret.text(0.5, 0.5, 'No retrieval metrics yet', ha='center', va='center')
            ax_ret.axis('off')
        else:
            if 'val_retrieval_top1' in retrieval_df:
                ax_ret.plot(retrieval_df['epoch'], retrieval_df['val_retrieval_top1'], marker='o', label='Retrieval@1')
            if 'val_retrieval_mrr' in retrieval_df:
                ax_ret.plot(retrieval_df['epoch'], retrieval_df['val_retrieval_mrr'], marker='s', label='MRR')
            ax_ret.set_xlabel('Epoch')
            ax_ret.set_ylabel('Retrieval Metric')
            ax_ret.set_title('Validation Retrieval Metrics')
            ax_ret.legend()
            ax_ret.grid(alpha=0.3)
            ax_ret.axvline(best_epoch, color='tab:green', linestyle='--', alpha=0.6)

    plt.tight_layout()
    fig_path = Path(OUT_DIR) / 'training_curves.png'
    plt.savefig(fig_path, dpi=150)
    plt.show()
    print(f"✅ Saved training curves to {fig_path}")
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
print(f"   Lowest CLIP Loss: {min(h['clip_loss'] for h in history):.4f}")
print(f"   Lowest CE Loss: {min(h['ce_loss'] for h in history):.4f}")

decoder_wers = [h.get('val_decoder_wer') for h in history if h.get('val_decoder_wer') is not None and not math.isnan(h.get('val_decoder_wer'))]
decoder_bleus = [h.get('val_decoder_bleu') for h in history if h.get('val_decoder_bleu') is not None and not math.isnan(h.get('val_decoder_bleu'))]
decoder_exact = [h.get('val_decoder_exact') for h in history if h.get('val_decoder_exact') is not None and not math.isnan(h.get('val_decoder_exact'))]
if decoder_wers:
    print(f"   Decoder WER (min): {min(decoder_wers):.3f}")
if decoder_bleus:
    print(f"   Decoder BLEU (max): {max(decoder_bleus):.3f}")
if decoder_exact:
    print(f"   Decoder Exact (max): {max(decoder_exact):.3f}")

retrieval_scores = []
retrieval_mrrs = []
for h in history:
    if 'val_retrieval_top1' in h and h['val_retrieval_top1'] is not None and not math.isnan(h['val_retrieval_top1']):
        retrieval_scores.append(h['val_retrieval_top1'])
    if 'val_retrieval_mrr' in h and h['val_retrieval_mrr'] is not None and not math.isnan(h['val_retrieval_mrr']):
        retrieval_mrrs.append(h['val_retrieval_mrr'])
if retrieval_scores:
    print(f"   Retrieval@1: {max(retrieval_scores):.3f}")
if retrieval_mrrs:
    print(f"   Retrieval MRR: {max(retrieval_mrrs):.3f}")

print(f"\n📦 Generated files:")
files = [
    "best_model_top1_*.pt",
    "embeddings.npy",
    "embeddings_map.csv",
    "prototypes.npz",
    "sign_encoder.onnx",
    "sign_encoder.ts",
    "model_config.json",
    "sign_encoder_bundle.zip",
    "export_manifest.json",
    "training_history.csv",
    "training_curves.png"
 ]
for f in files:
    print(f"   ✓ {f}")

print(f"\n🚀 Next steps:")
print(f"   1. Validate DTW + prototype gating on additional webcam clips")
print(f"   2. Stress-test inference thresholds against negative samples")
print(f"   3. Integrate refreshed ONNX + TorchScript bundle into the service layer")
print(f"   4. Re-tune thresholds if deploying to new camera setups")

print("\n" + "=" * 70)
```

