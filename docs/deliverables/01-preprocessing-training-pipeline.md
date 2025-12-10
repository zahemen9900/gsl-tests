# GhSL Sign2Text: Preprocessing & Training Pipeline
**Version:** 1.0  
**Status:** ✅ Production-Ready  
**Last Updated:** December 2024

---

## Executive Summary

This document describes the complete preprocessing and training pipeline for the **GhSL Sign2Text Translation System**. The system converts sign language videos into text through a multi-stage pipeline that includes landmark extraction, feature normalization, data augmentation, and multi-objective transformer training.

### Key Achievements

| Metric | Value |
|--------|-------|
| Instance Top-1 Accuracy | ~74.7% |
| Positive Embedding Similarity | > 0.85 |
| Negative Embedding Similarity | < 0.45 |
| Model Parameters | ~2.5M |
| Feature Dimension | 540 per frame |
| Sequence Length | 64 frames |

---

## 1. Preprocessing Pipeline

### 1.1 Feature Extraction Overview

The preprocessing stage converts raw sign language videos into normalized pose sequences suitable for training.

```
Raw Video → MediaPipe Holistic → Landmark Extraction → Torso Normalization → Quality Gating → .npy Output
```

### 1.2 Landmark Configuration

Features are extracted using MediaPipe Holistic with the following structure:

| Body Part | Landmarks | Features per Landmark | Total Features |
|-----------|-----------|----------------------|----------------|
| Pose | 33 | 4 (x, y, z, visibility) | 132 |
| Left Hand | 21 | 3 (x, y, z) | 63 |
| Right Hand | 21 | 3 (x, y, z) | 63 |
| Face (downsampled) | 94 | 3 (x, y, z) | 282 |
| **Total** | **169** | - | **540** |

**Face Downsampling:** Every 5th face landmark is used (94 out of 468) to reduce dimensionality while retaining expressive information.

### 1.3 Torso-Centric Normalization

All landmarks are normalized relative to the torso to reduce camera variance:

```python
def normalize_frame_landmarks(frame_feats):
    # Reference points for torso
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    # Compute torso center
    torso_pts = pose_coords[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]
    torso_center = torso_pts.mean(axis=0)
    
    # Center all coordinates
    pose_coords -= torso_center
    left_hand -= torso_center
    right_hand -= torso_center
    face -= torso_center
    
    # Compute scale from torso dimensions
    shoulder_span = np.linalg.norm(pose_coords[LEFT_SHOULDER] - pose_coords[RIGHT_SHOULDER])
    hip_span = np.linalg.norm(pose_coords[LEFT_HIP] - pose_coords[RIGHT_HIP])
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    scale = np.median([shoulder_span, hip_span, torso_height])
    
    # Scale-normalize all coordinates
    all_coords /= scale
```

### 1.4 Quality Gating Thresholds

Videos are filtered based on quality metrics:

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `MIN_FRAMES` | 48 | Minimum usable frames after filtering |
| `MIN_VALID_RATIO` | 0.45 | Minimum landmark detection ratio |
| `VISIBILITY_THRESHOLD` | 0.55 | Pose landmark visibility minimum |
| `MOTION_KEEP_THRESHOLD` | 1e-3 | Motion energy to retain frame |
| `MOTION_REJECTION_THRESHOLD` | 7.5e-4 | Clip rejection threshold |

### 1.5 Motion Energy Computation

Per-frame motion energy is computed to filter static frames and assess clip quality:

```python
def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    # Extract pose XYZ, hands, and face coordinates
    pose_xyz = frames[:, :33*4].reshape(frames.shape[0], 33, 4)[..., :3]
    hands = frames[:, 33*4:33*4 + 2*21*3]
    face = frames[:, 33*4 + 2*21*3:]
    coords = np.concatenate([pose_xyz.reshape(N, -1), hands, face], axis=1)
    
    # Compute frame-to-frame differences
    diffs = np.diff(coords, axis=0, prepend=coords[:1])
    return np.linalg.norm(diffs, axis=1)
```

### 1.6 Temporal Smoothing

Exponential Moving Average (EMA) smoothing reduces jitter:

```python
def exponential_smooth(sequence, alpha=0.2):
    smoothed = sequence.copy()
    for idx in range(1, sequence.shape[0]):
        smoothed[idx] = alpha * sequence[idx] + (1 - alpha) * smoothed[idx - 1]
    return smoothed
```

### 1.7 Global Statistics

After preprocessing, global mean and standard deviation are computed across all clips:

```python
np.savez(
    "proc/global_stats.npz",
    feature_mean=feature_mean.astype(np.float32),
    feature_std=feature_std.astype(np.float32),
    seq_lengths=np.array(seq_lengths, dtype=np.int32),
    motion_means=np.array(motion_means, dtype=np.float32)
)
```

### 1.8 Text Augmentation

The Gemini API is used to generate sentence variations for data augmentation:

```python
# Generate 5 variations per sentence
prompt = f"""
Generate 5 distinct, natural-sounding English variations of the following 
sign language gloss/sentence. Keep the meaning identical but vary the phrasing.
Input: "{sentence}"
"""
# Stored in proc/sentence_variations.json
```

---

## 2. Model Architecture

### 2.1 Architecture Overview

```
Visual Input (B, T, 540)
        ↓
  FrameProjector → (B, T, 160)
        ↓
TemporalTransformerEncoder → Pooled Embedding (B, 256) + Context (B, T, 256)
        ↓                              ↓
  SupCon Loss              TransformerDecoder → Text Output
        ↓                              ↓
  Masked Frame Loss            CrossEntropy Loss
```

### 2.2 FrameProjector

Projects 540-dim pose features to a compact representation:

```python
class FrameProjector(nn.Module):
    def __init__(self, in_dim=540, proj_dim=160):
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
```

### 2.3 TemporalTransformerEncoder

Full Transformer encoder with sinusoidal positional encoding:

```python
class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim=160,
        embed_dim=256,
        n_layers=4,
        attn_heads=4,
        ff_dim=512,
        dropout=0.1
    ):
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.pos_encoder = TemporalPositionalEncoding(embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attn_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

**Key Features:**
- **Pre-norm architecture** (`norm_first=True`) for training stability
- **Mean pooling** over time dimension for contrastive embeddings
- **L2 normalization** of output embeddings
- Returns both pooled embedding and full sequence context for decoder

### 2.4 Text Decoder

Auto-regressive Transformer decoder:

```python
class SignTranslationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, attn_heads=4):
        # Token embedding + positional encoding
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # 2-layer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=attn_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.generator = nn.Linear(embed_dim, vocab_size)
```

### 2.5 Masked Frame Modeling Head

MAE-style self-supervised objective:

```python
def masked_frame_loss(self, src, mask_ratio=0.15):
    proj = self.projector(src)
    
    # Random mask 15% of frames
    mask = (torch.rand(bsz, seq_len) < mask_ratio)
    masked_proj = proj.clone()
    masked_proj[mask] = self.mask_token.expand_as(proj)[mask]
    
    # Reconstruct masked frames
    _, _, context = self.encoder(masked_proj, return_sequence=True)
    recon = self.mask_decoder(context)
    return F.smooth_l1_loss(recon[mask], proj[mask])
```

---

## 3. Training Configuration

### 3.1 Hyperparameters

```python
CFG = {
    # Data
    'seq_len': 64,
    'input_dim': 540,
    'batch_size': 8,
    'val_split': 0.15,
    
    # Architecture
    'proj_dim': 160,
    'embed_dim': 256,
    'attn_heads': 4,
    'encoder_layers': 4,
    'encoder_ff_dim': 512,
    'encoder_dropout': 0.1,
    
    # Training
    'epochs': 100,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    'label_smoothing': 0.1,
    
    # Multi-Objective Weights
    'temperature': 0.07,          # SupCon temperature
    'masked_frame_ratio': 0.15,
    'mask_loss_weight': 0.2,
    
    # Beam Search
    'beam_size': 4,
    'beam_length_penalty': 0.8,
    'decoder_max_len': 64,
    
    # Augmentation
    'time_warp': 0.15,
    'frame_dropout': 0.12,
    'noise_sigma': 0.012,
    'temporal_shift': 4,
}
```

### 3.2 Multi-Objective Loss Function

The training combines three complementary objectives:

$$\mathcal{L}_{total} = \lambda \cdot \mathcal{L}_{SupCon} + (1-\lambda) \cdot \mathcal{L}_{CE} + \gamma \cdot \mathcal{L}_{mask}$$

Where:
- $\lambda = 0.5$ (balances contrastive vs translation)
- $\gamma = 0.2$ (mask loss weight)

**Supervised Contrastive Loss (SupConLoss):**
- Clusters embeddings by sentence_id
- Temperature scaling: τ = 0.07
- Uses concatenated dual-view embeddings

**CrossEntropy Loss:**
- Teacher forcing with label smoothing (0.1)
- Ignores padding tokens

**Masked Frame Modeling Loss:**
- 15% of frames randomly masked
- Smooth L1 reconstruction loss

### 3.3 Data Augmentation Pipeline

| Augmentation | Training | Validation |
|--------------|----------|------------|
| Time Warp | 0.15 | 0.05 |
| Frame Dropout | 0.12 | 0.06 |
| Noise σ | 0.012 | 0.006 |
| Temporal Shift | ±4 frames | ±2 frames |
| Y-Axis Rotation | ±15° | 0° |

**Dual-View Generation:**
```python
def split_dual_views(seq):
    # Even/odd frame splitting
    view1 = seq[0::2]  # Even frames
    view2 = seq[1::2]  # Odd frames
    
    # Apply rotation to view2 for viewpoint variation
    view2 = apply_y_rotation(view2, random_angle)
    return view1, view2
```

### 3.4 ClassBalancedSampler

Ensures effective SupCon learning:

```python
class ClassBalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, instances_per_class=2):
        # Each batch contains 2 instances of each sentence_id
        self.classes_per_batch = batch_size // instances_per_class
```

---

## 4. Training Loop

### 4.1 Training Epoch

```python
def train_epoch(model, loader, optimizer, supcon_criterion, ce_criterion, lambda_val=0.5):
    for batch in loader:
        a, b, labels, sentence_ids, text_a, text_b, token_ids = batch
        
        # 1. Supervised Contrastive Loss
        emb_a = model.encode_visual(a)
        emb_b = model.encode_visual(b)
        features = torch.cat([emb_a, emb_b], dim=0)
        combined_labels = torch.cat([sentence_ids, sentence_ids], dim=0)
        loss_supcon = supcon_criterion(features, combined_labels)
        
        # 2. CrossEntropy Loss (Seq2Seq)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        logits, _ = model(a, tgt_input, tgt_mask=tgt_mask)
        loss_ce = ce_criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
        
        # 3. Masked Frame Modeling
        loss_mask = model.masked_frame_loss(a, mask_ratio=0.15)
        
        # Combined loss
        loss = lambda_val * loss_supcon + (1 - lambda_val) * loss_ce + 0.2 * loss_mask
```

### 4.2 Evaluation Metrics

**Contrastive Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| Inst@1 | Instance retrieval accuracy | > 0.70 |
| PosSim | Cosine similarity of positive pairs | > 0.85 |
| NegSim | Mean similarity of negatives | < 0.50 |

**Translation Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| WER | Word Error Rate | < 30% |
| BLEU-4 | N-gram precision | > 20.0 |
| Exact Match | Perfect predictions | > 10% |

---

## 5. Export Pipeline

### 5.1 Model Export

```python
# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "sign_encoder.onnx",
    input_names=['visual_input'],
    output_names=['embedding'],
    dynamic_axes={
        'visual_input': {0: 'batch', 1: 'frames'},
        'embedding': {0: 'batch'}
    },
    opset_version=18
)
```

### 5.2 Exported Artifacts

| File | Description |
|------|-------------|
| `sign_encoder.onnx` | ONNX model for inference |
| `embeddings.npy` | Reference embeddings database (N, 256) |
| `embeddings_map.csv` | Metadata mapping |
| `prototypes.npz` | Sentence-level prototype embeddings |
| `global_stats.npz` | Feature normalization statistics |
| `model_config.json` | Training configuration snapshot |
| `tokenizer_vocab.json` | Word-to-ID vocabulary mapping |

---

## 6. File Structure

```
gsl-tests/
├── preprocessing.ipynb       # Feature extraction pipeline
├── training.ipynb            # Model training pipeline
├── proc/
│   ├── processed_metadata.csv
│   ├── global_stats.npz
│   ├── sentence_variations.json
│   └── low_quality_videos.csv
├── features/
│   └── pose_data/           # .npy feature files
└── runs/
    ├── best_model_*.pt      # Training checkpoints
    ├── embeddings.npy
    ├── embeddings_map.csv
    ├── prototypes.npz
    ├── tokenizer_vocab.json
    └── training_history.csv
```

---

## 7. Future Improvements

- [ ] Subword tokenization (BPE/SentencePiece) for better OOV handling
- [ ] Conformer encoder for local + global attention
- [ ] CTC auxiliary loss for alignment learning
- [ ] Multi-task learning with gloss prediction
- [ ] Knowledge distillation from larger models
- [ ] Curriculum learning based on motion complexity
