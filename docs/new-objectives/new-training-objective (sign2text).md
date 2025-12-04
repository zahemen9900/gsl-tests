# Technical Spec: GhSL Sign2Text Translation System
**Version:** 2.0  
**Status:** ✅ Implemented  
**Target:** Real-Time Pose-Based Sign Language Translation (Sign2Text)  
**Architecture:** Transformer Encoder-Decoder with Multi-Objective Training

---

## 1. Executive Summary

The GhSL Sign2Text system implements a **Sequence-to-Sequence translation model** that converts pose landmark sequences directly into text. The architecture combines:

- **Transformer-based visual encoder** (4 layers, 4 attention heads)
- **Auto-regressive Transformer decoder** (2 layers)
- **Multi-objective training**: SupCon + CrossEntropy + Masked Frame Modeling
- **Beam search decoding** with length penalty for inference

### Key Achievements
| Metric | Value |
|--------|-------|
| Instance Top-1 Accuracy | ~74.7% |
| Positive Similarity | > 0.85 |
| Negative Similarity | < 0.45 |
| Model Parameters | ~2.5M |

---

## 2. Model Architecture

### 2.1 Visual Encoder Pipeline

```
Input: (B, T, 540)  →  FrameProjector  →  (B, T, 160)  →  TemporalTransformerEncoder  →  (B, T, 256) + pooled embedding
```

#### FrameProjector
Projects per-frame 540-dim pose features to a lower-dimensional representation:
```python
class FrameProjector(nn.Module):
    def __init__(self, in_dim, proj_dim=160):
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)
```

#### TemporalTransformerEncoder
Full Transformer encoder replacing the previous GRU-based approach:
```python
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=256, n_layers=4, attn_heads=4, ff_dim=512, dropout=0.1):
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.pos_encoder = TemporalPositionalEncoding(embed_dim, dropout, max_len=256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attn_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
```

**Key features:**
- **Sinusoidal positional encoding** for temporal awareness
- **Pre-norm architecture** (`norm_first=True`) for training stability
- **GELU activation** throughout
- **Mean pooling** over time dimension for contrastive embeddings
- **L2 normalization** of output embeddings

### 2.2 Text Decoder

Auto-regressive Transformer decoder for text generation:
```python
# Decoder components
self.tgt_embed = nn.Embedding(vocab_size, embed_dim)
self.pos_encoder = PositionalEncoding(embed_dim)
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

### 2.3 Masked Frame Modeling Head (Auxiliary)

MAE-style self-supervised objective on the visual encoder:
```python
self.mask_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
self.mask_decoder = nn.Sequential(
    nn.LayerNorm(embed_dim),
    nn.Linear(embed_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, proj_dim)
)
```

---

## 3. Training Configuration

### 3.1 Hyperparameters (CFG)

```python
CFG = {
    # Data
    'seq_len': 64,
    'input_dim': 540,
    'batch_size': 32,
    'val_split': 0.15,
    
    # Architecture
    'proj_dim': 160,
    'embed_dim': 256,
    'attn_heads': 4,
    'encoder_layers': 4,
    'encoder_ff_dim': 512,
    'encoder_dropout': 0.1,
    
    # Training
    'epochs': 50,
    'lr': 1e-4,
    'label_smoothing': 0.1,
    
    # Auxiliary loss
    'masked_frame_ratio': 0.15,
    'mask_loss_weight': 0.2,
    
    # Beam search
    'beam_size': 4,
    'beam_length_penalty': 0.8,
    'decoder_max_len': 64,
    
    # Augmentation
    'time_warp': 0.15,
    'frame_dropout': 0.1,
    'noise_sigma': 0.02,
    'temporal_shift': 3,
}
```

### 3.2 Multi-Objective Loss Function

The training combines three complementary objectives:

$$\mathcal{L}_{total} = \lambda \cdot \mathcal{L}_{SupCon} + (1-\lambda) \cdot \mathcal{L}_{CE} + \gamma \cdot \mathcal{L}_{mask}$$

Where:
- $\lambda = 0.5$ (balances contrastive vs translation)
- $\gamma = 0.2$ (mask loss weight)

#### Supervised Contrastive Loss (SupConLoss)
```python
class SupConLoss(nn.Module):
    """https://arxiv.org/abs/2004.11362"""
    def __init__(self, temperature=0.07):
        self.temperature = temperature
```
- **Purpose:** Cluster embeddings by semantic meaning (sentence_id)
- **Input:** Concatenated dual-view embeddings `[emb_a, emb_b]` with labels
- **Effect:** Same-sign videos pulled together, different signs pushed apart

#### CrossEntropy Loss
```python
ce_criterion = nn.CrossEntropyLoss(
    ignore_index=pad_token_id,
    label_smoothing=0.1
)
```
- **Purpose:** Train decoder to predict text tokens
- **Input:** Decoder logits vs shifted target tokens (teacher forcing)
- **Label smoothing:** 0.1 to improve generalization

#### Masked Frame Modeling Loss
```python
def masked_frame_loss(self, src, mask_ratio=0.15):
    # Randomly mask 15% of frames
    mask = (torch.rand(bsz, seq_len, device=proj.device) < mask_ratio)
    masked_proj = proj.clone()
    masked_proj[mask] = self.mask_token.expand_as(proj)[mask]
    # Reconstruct masked frames
    _, _, context = self.encoder(masked_proj, return_sequence=True)
    recon = self.mask_decoder(context)
    loss = F.smooth_l1_loss(recon[mask], proj[mask])
```
- **Purpose:** Self-supervised regularization, learns temporal dynamics
- **Mask ratio:** 15% of frames randomly masked
- **Reconstruction:** Smooth L1 loss on projector outputs

---

## 4. Data Pipeline

### 4.1 Dual-View Generation

The `SignDataset` generates two augmented views from each sample:

```python
def _split_dual_views(self, seq):
    """Split sequence into even/odd frame views."""
    view1 = seq[0::2]  # Even frames
    view2 = seq[1::2]  # Odd frames
    
    # Apply Y-axis rotation to view2
    angle = np.random.uniform(-self.cfg['rotation_angle'], self.cfg['rotation_angle'])
    view2 = self._apply_y_rotation(view2, np.radians(angle))
    
    return view1, view2
```

**Y-Axis Rotation Mathematics:**
$$x' = x \cos(\theta) + z \sin(\theta)$$
$$z' = -x \sin(\theta) + z \cos(\theta)$$

### 4.2 Text Variation Sampling

Each sample can have multiple valid text translations stored in the metadata:
```python
def _sample_text_variation(self, row):
    variations = [row['sentence']]
    if pd.notna(row.get('sentence_variations')):
        variations.extend(row['sentence_variations'].split('|'))
    return random.choice(variations)
```

### 4.3 ClassBalancedSampler

Ensures every batch contains K instances of each sentence class for effective SupCon:
```python
class ClassBalancedSampler(Sampler):
    def __init__(self, meta_df, batch_size, instances_per_class=2):
        self.instances_per_class = instances_per_class
        self.classes_per_batch = batch_size // instances_per_class
```

### 4.4 Augmentation Pipeline

| Augmentation | Training | Validation |
|--------------|----------|------------|
| Time Warp | 0.15 | 0.05 |
| Frame Dropout | 0.10 | 0.05 |
| Noise σ | 0.02 | 0.01 |
| Temporal Shift | ±3 frames | ±1 frame |
| Y-Axis Rotation | ±15° | 0° |

### 4.5 SimpleTokenizer

Word-level tokenizer with special tokens:
```python
class SimpleTokenizer:
    def __init__(self, sentences, vocab_size=1000):
        self.pad_token, self.pad_token_id = '<PAD>', 0
        self.sos_token, self.sos_token_id = '<SOS>', 1
        self.eos_token, self.eos_token_id = '<EOS>', 2
        self.unk_token, self.unk_token_id = '<UNK>', 3
        # Build vocabulary from word frequencies
```

---

## 5. Inference: Beam Search Decoding

### 5.1 Algorithm

```python
def beam_search_decode_batch(
    model, src_batch, tokenizer, device,
    max_len=64, beam_size=4, length_penalty=0.8
):
    # Encode visual sequence
    proj = model.projector(src_batch)
    _, _, context = model.encoder(proj, return_sequence=True)
    memory = context.permute(1, 0, 2)  # (T, B, E)
    
    # Beam search per sample
    for idx in range(batch_size):
        best_seq = _beam_search_single(model, memory[:, idx:idx+1, :], ...)
```

### 5.2 Length Penalty

Prevents overly short predictions:
$$score_{adjusted} = \frac{score}{((5 + length) / 6)^\alpha}$$

Where $\alpha = 0.8$ (CFG `beam_length_penalty`)

### 5.3 Streaming Inference

For real-time applications, the system supports streaming with motion gating:

```python
class StreamingDecoder:
    def __init__(self, model, tokenizer, cfg):
        self.buffer = []
        self.motion_threshold = cfg['motion_floor']
    
    def process_frame(self, frame):
        self.buffer.append(frame)
        if len(self.buffer) >= self.min_frames:
            motion = compute_motion_energy(self.buffer)
            if motion.mean() > self.motion_threshold:
                return self.decode_buffer()
        return None
```

---

## 6. Evaluation Metrics

### 6.1 Contrastive Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Inst@1** | Instance-level retrieval accuracy | > 0.70 |
| **PosSim** | Cosine similarity of positive pairs | > 0.85 |
| **NegSim** | Mean similarity of negatives | < 0.50 |

### 6.2 Translation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **WER** | Word Error Rate $(S+D+I)/N$ | < 30% |
| **BLEU-4** | 4-gram precision with brevity penalty | > 20.0 |
| **Exact Match** | Percentage of perfect predictions | > 10% |

### 6.3 Retrieval Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Ret@1** | Sentence retrieval top-1 accuracy | > 0.65 |
| **MRR** | Mean Reciprocal Rank | > 0.75 |

---

## 7. Export Pipeline

### 7.1 ONNX Export

```python
torch.onnx.export(
    model,
    dummy_input,
    "sign_encoder.onnx",
    input_names=['visual_input'],
    output_names=['embedding', 'attention'],
    dynamic_axes={
        'visual_input': {0: 'batch', 1: 'frames'},
        'embedding': {0: 'batch'}
    },
    opset_version=18
)
```

### 7.2 Exported Artifacts

| File | Description |
|------|-------------|
| `sign_encoder.onnx` | ONNX model for CPU/GPU inference |
| `embeddings.npy` | Reference embeddings database |
| `embeddings_map.csv` | Metadata mapping for retrieval |
| `prototypes.npz` | Sentence-level prototype embeddings |
| `global_stats.npz` | Feature normalization statistics |
| `model_config.json` | Training configuration snapshot |

---

## 8. Implementation Notes

### 8.1 Training Loop Structure

```python
def train_epoch(model, loader, optimizer, supcon_criterion, ce_criterion, lambda_val=0.5):
    for a, b, labels, sentence_ids, text_a, text_b, token_ids in loader:
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

### 8.2 Key Design Decisions

1. **Pre-norm Transformer**: Uses `norm_first=True` for better training stability
2. **Dual-view splitting**: Even/odd frames ensure temporal coverage while creating distinct views
3. **Y-axis rotation**: Simulates viewpoint changes without expensive 3D rendering
4. **ClassBalancedSampler**: Critical for effective SupCon with imbalanced data
5. **Beam search with length penalty**: Balances fluency vs brevity in outputs
6. **Masked frame modeling**: Self-supervised regularization that improves encoder representations

### 8.3 Future Improvements

- [ ] Subword tokenization (BPE/SentencePiece) for better OOV handling
- [ ] Conformer encoder for local + global attention
- [ ] CTC auxiliary loss for alignment learning
- [ ] Multi-task learning with gloss prediction
- [ ] Knowledge distillation from larger models