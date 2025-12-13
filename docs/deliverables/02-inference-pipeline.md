# GhSL Sign2Text: Inference Pipeline
**Version:** 1.0  
**Status:** ✅ Production-Ready  
**Last Updated:** December 2024

---

## Executive Summary

This document describes the complete inference pipeline for the **GhSL Sign2Text Translation System**, covering both real-time streaming inference and batch processing modes. The system supports two inference approaches: direct Seq2Seq text generation via beam search decoding, and embedding-based nearest-neighbor retrieval as a fallback mechanism.

### Key Performance Metrics

| Metric | Target |
|--------|--------|
| Encoding Latency | < 50ms |
| Beam Search Latency | < 100ms |
| Total Inference | < 200ms |
| Supported Providers | CPU, CUDA |

---

## 1. Inference Modes

### 1.1 Translation Mode (Primary)

Direct text generation using beam search decoding:

```
Visual Input → FrameProjector → TransformerEncoder → TransformerDecoder → Beam Search → Text Output
```

**When to Use:**
- Real-time translation requirements
- Novel/unseen sign combinations
- Sentence-level output needed

### 1.2 Retrieval Mode (Fallback)

Embedding-based nearest-neighbor matching:

```
Visual Input → Encoder → Embedding → Cosine Similarity → Top-K Results → Optional DTW Re-ranking
```

**When to Use:**
- High-confidence phrase matching
- Vocabulary-constrained scenarios
- Debugging embedding quality

---

## 2. Beam Search Decoding

### 2.1 Algorithm Overview

```python
def beam_search_decode_batch(
    model,
    src_batch: torch.Tensor,
    tokenizer,
    device,
    max_len: int = 64,
    beam_size: int = 4,
    length_penalty: float = 0.8
) -> List[List[int]]:
    # Encode visual sequence
    proj = model.projector(src_batch)
    _, _, context = model.encoder(proj, return_sequence=True)
    memory = context.permute(1, 0, 2)  # (T, B, E)
    
    # Beam search per sample
    decoded = []
    for idx in range(src_batch.size(0)):
        sample_memory = memory[:, idx:idx + 1, :]
        best_seq = _beam_search_single(
            model, sample_memory, tokenizer, device,
            max_len=max_len, beam_size=beam_size, length_penalty=length_penalty
        )
        decoded.append(best_seq)
    return decoded
```

### 2.2 Single-Sample Beam Search

```python
def _beam_search_single(model, memory, tokenizer, device, max_len, beam_size, length_penalty):
    bos = tokenizer.sos_token_id
    eos = tokenizer.eos_token_id
    beams = [(0.0, [bos], False)]  # (log_prob, sequence, is_done)
    finished = []

    for step in range(max_len):
        new_beams = []
        for logprob, seq, done in beams:
            if done:
                new_beams.append((logprob, seq, True))
                continue
            
            # Decode next token
            tgt = torch.tensor(seq, device=device).unsqueeze(0)
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
            tgt_emb = model.tgt_embed(tgt).permute(1, 0, 2)
            tgt_emb = model.pos_encoder(tgt_emb)
            decoder_out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.generator(decoder_out[-1].unsqueeze(0))
            log_probs = torch.log_softmax(logits.squeeze(0), dim=-1)
            
            # Expand beam with top-k tokens
            topk = torch.topk(log_probs, k=beam_size)
            for value, index in zip(topk.values.tolist(), topk.indices.tolist()):
                next_seq = seq + [index]
                next_done = index == eos
                next_logprob = logprob + value
                new_beams.append((next_logprob, next_seq, next_done))
                if next_done:
                    finished.append((next_logprob, next_seq))
        
        # Prune to top-k beams
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
        
        if all(done for _, _, done in beams):
            break
    
    return _select_best_with_length_penalty(finished or beams, length_penalty)
```

### 2.3 Length Penalty

Prevents overly short predictions:

$$score_{adjusted} = \frac{score}{((5 + length) / 6)^\alpha}$$

Where α = 0.8 (configurable via `beam_length_penalty`)

### 2.4 Configuration

```python
CFG = {
    'beam_size': 4,
    'beam_length_penalty': 0.8,
    'decoder_max_len': 64,
}
```

---

## 3. Streaming Inference

### 3.1 Motion-Gated Flow

For real-time webcam applications, the system uses motion gating to determine when to trigger inference:

```python
class StreamingDecoder:
    def __init__(self, model, tokenizer, cfg):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.buffer = []
        self.min_frames = cfg.get('min_frames', 16)
        self.max_frames = cfg['seq_len']  # 64
        self.motion_threshold = cfg['motion_floor']  # 7.5e-4
    
    def process_frame(self, frame_landmarks):
        """Process incoming frame and optionally trigger inference."""
        self.buffer.append(frame_landmarks)
        
        # Wait for minimum frames
        if len(self.buffer) < self.min_frames:
            return {'status': 'buffering', 'frames': len(self.buffer)}
        
        # Check motion energy
        motion = self._compute_motion_energy()
        if motion < self.motion_threshold:
            return {'status': 'waiting', 'reason': 'low_motion'}
        
        # Trigger inference when buffer is full
        if len(self.buffer) >= self.max_frames:
            return self._decode_buffer()
        
        return {'status': 'buffering', 'frames': len(self.buffer)}
    
    def _decode_buffer(self):
        """Run beam search on buffer and clear."""
        seq = self._preprocess_buffer()
        with torch.no_grad():
            tokens = beam_search_decode(
                self.model, seq, self.tokenizer,
                beam_size=4, length_penalty=0.8
            )
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.buffer.clear()
        return {'status': 'decoded', 'text': text}
```

### 3.2 Motion Energy Computation

```python
def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    
    # Extract pose XYZ (ignoring visibility)
    pose_xyz = frames[:, :33*4].reshape(frames.shape[0], 33, 4)[..., :3]
    pose_xyz = pose_xyz.reshape(frames.shape[0], -1)
    
    # Concatenate hands and face
    hands = frames[:, 33*4:33*4 + 2*21*3]
    face = frames[:, 33*4 + 2*21*3:]
    coords = np.concatenate([pose_xyz, hands, face], axis=1)
    
    # Frame-to-frame differences
    diffs = np.diff(coords, axis=0, prepend=coords[:1])
    return np.linalg.norm(diffs, axis=1).astype(np.float32)
```

### 3.3 Streaming Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `STREAM_WINDOW` | 64 | Frames per inference window |
| `STREAM_BUFFER_LIMIT` | 160 | Max buffer size before trimming |
| `STREAM_COOLDOWN_MS` | 2000 | Minimum time between inferences |
| `motion_floor` | 7.5e-4 | Minimum motion for valid clip |

---

## 4. Retrieval-Based Inference

### 4.1 Embedding Extraction

```python
def encode_visual(model, x):
    """Extract normalized embedding for retrieval."""
    proj = model.projector(x)
    embedding, _ = model.encoder(proj)
    return F.normalize(embedding, dim=-1)
```

### 4.2 Nearest-Neighbor Scoring

```python
def score_neighbours(query_embedding, reference_embeddings):
    # Cosine similarity (embeddings are L2-normalized)
    raw_scores = reference_embeddings @ query_embedding
    
    # Softmax for confidence
    logits = raw_scores / confidence_temperature
    probs = softmax(logits)
    
    return raw_scores, probs
```

### 4.3 DTW Re-ranking (Optional)

For improved accuracy, top-K candidates can be re-ranked using Dynamic Time Warping:

```python
def dtw_rerank(query_seq, candidates, cfg):
    refined_scores = raw_scores.copy()
    
    for idx in top_k_candidates:
        ref_seq = load_reference_sequence(idx)
        
        # Compute DTW distance
        dist, _ = fastdtw(
            query_seq, ref_seq,
            radius=cfg['dtw_radius'],  # 6
            dist=lambda x, y: np.linalg.norm(x - y)
        )
        
        # Convert distance to score
        dtw_score = np.exp(-cfg['dtw_lambda'] * dist)  # lambda=0.002
        
        # Blend cosine and DTW scores
        refined_scores[idx] = (
            cfg['dtw_alpha'] * raw_scores[idx] +  # alpha=0.65
            (1 - cfg['dtw_alpha']) * dtw_score
        )
    
    return refined_scores
```

### 4.4 Acceptance Gating

```python
def apply_gating(scores, probs, top1_sentence_id, prototype_lookup, embedding, cfg):
    top1_score = scores[0]
    top2_score = scores[1] if len(scores) > 1 else -1.0
    
    # Check similarity threshold
    similarity_ok = top1_score >= cfg['sim_threshold']  # 0.58
    
    # Check margin between top-1 and top-2
    margin_ok = (top1_score - top2_score) >= cfg['sim_margin']  # 0.10
    
    # Check prototype agreement
    proto_ok = True
    if top1_sentence_id in prototype_lookup:
        proto_sim = np.dot(prototype_lookup[top1_sentence_id], embedding)
        proto_ok = proto_sim >= (cfg['sim_threshold'] - 0.05)
    
    # All checks must pass
    accepted = similarity_ok and margin_ok and proto_ok
    
    if not accepted:
        if not similarity_ok:
            reason = 'low_similarity'
        elif not margin_ok:
            reason = 'ambiguous_top2'
        else:
            reason = 'prototype_disagreement'
    else:
        reason = 'ok'
    
    return accepted, reason
```

---

## 5. Preprocessing Pipeline (Inference)

### 5.1 Sequence Preparation

```python
def prepare_sequence(seq: np.ndarray, global_mean, global_std, seq_len=64) -> np.ndarray:
    data = seq.astype(np.float32, copy=True)
    
    # Validate
    if data.shape[0] == 0:
        raise ValueError("Sequence contains no frames")
    
    # Center crop if too long
    if data.shape[0] > seq_len:
        data = center_time_crop(data, seq_len)
    
    # Apply global normalization
    data = (data - global_mean) / (global_std + 1e-6)
    
    # Pad if too short (repeat last frame)
    if data.shape[0] < seq_len:
        pad = np.repeat(data[-1:], seq_len - data.shape[0], axis=0)
        data = np.vstack([data, pad])
    
    return data
```

### 5.2 Motion Gating

```python
def validate_motion(frames, motion_floor=7.5e-4):
    motion = compute_motion_energy(frames)
    motion_mean = float(motion.mean())
    
    if motion_mean < motion_floor:
        raise ValueError(
            f"Insufficient motion ({motion_mean:.4e} < {motion_floor:.4e}); clip rejected."
        )
    
    return motion_mean
```

---

## 6. ONNX Runtime Inference

### 6.1 Session Initialization

```python
class OnnxSignInferenceService:
    def __init__(self, settings):
        # Configure providers
        if settings.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        # Session options for performance
        session_opts = ort.SessionOptions()
        session_opts.enable_mem_pattern = False
        session_opts.enable_cpu_mem_arena = True
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create session
        self._session = ort.InferenceSession(
            str(settings.onnx_model_path),
            providers=providers,
            sess_options=session_opts,
        )
        
        # Get input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
```

### 6.2 Running Inference

```python
def run_model(self, sequence: np.ndarray) -> np.ndarray:
    """Run ONNX model and return normalized embedding."""
    with self._lock:  # Thread safety
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: sequence}
        )
    
    embedding = outputs[0]
    if embedding.ndim == 2:
        embedding = embedding[0]
    
    # L2 normalize
    norm = np.linalg.norm(embedding) + 1e-8
    return embedding / norm
```

---

## 7. Inference Artifacts

### 7.1 Required Files

| File | Description | Location |
|------|-------------|----------|
| `sign_encoder.onnx` | ONNX model | `runs/` |
| `embeddings.npy` | Reference embeddings (N, 256) | `runs/` |
| `embeddings_map.csv` | Metadata mapping | `runs/` |
| `global_stats.npz` | Mean/std for normalization | `proc/` or `runs/` |
| `prototypes.npz` | Sentence prototypes (optional) | `runs/` |
| `tokenizer_vocab.json` | Token vocabulary | `runs/` |

### 7.2 Loading Artifacts

```python
def load_artifacts(settings):
    # Global stats
    stats = np.load(settings.global_stats_path)
    global_mean = stats['feature_mean']
    global_std = stats['feature_std']
    
    # Reference embeddings
    embeddings = np.load(settings.embeddings_path)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Mapping
    mapping = pd.read_csv(settings.mapping_path)
    
    # Prototypes (optional)
    prototypes = {}
    if settings.prototypes_path and Path(settings.prototypes_path).exists():
        proto_blob = np.load(settings.prototypes_path)
        for sid, emb in zip(proto_blob['sentence_ids'], proto_blob['embeddings']):
            prototypes[int(sid)] = emb / (np.linalg.norm(emb) + 1e-8)
    
    return global_mean, global_std, embeddings, mapping, prototypes
```

---

## 8. Evaluation Metrics

### 8.1 Translation Quality

```python
def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    
    # Dynamic programming edit distance
    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    # ... (standard Levenshtein computation)
    
    return float(dp[-1, -1] / max(len(ref_words), 1))

def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Calculate BLEU score with brevity penalty."""
    # N-gram precision for n=1,2,3,4
    # Brevity penalty
    # Geometric mean
    pass
```

### 8.2 Retrieval Quality

```python
def evaluate_retrieval(embeddings, sentence_ids):
    # Cosine similarity matrix
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    
    # Top-1 retrieval accuracy
    top1_hits = []
    reciprocal_ranks = []
    
    for idx in range(len(sentence_ids)):
        positives = np.where(sentence_ids == sentence_ids[idx])[0]
        positives = positives[positives != idx]  # Exclude self
        
        if positives.size == 0:
            continue
        
        order = np.argsort(-sims[idx])
        for rank, j in enumerate(order, start=1):
            if j in positives:
                top1_hits.append(1 if rank == 1 else 0)
                reciprocal_ranks.append(1.0 / rank)
                break
    
    return {
        'ret_top1': np.mean(top1_hits),
        'ret_mrr': np.mean(reciprocal_ranks),
    }
```

---

## 9. Configuration Reference

### 9.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEQ_LEN` | 64 | Nominal sequence length |
| `FEATURE_DIM` | 540 | Input feature dimension |
| `MOTION_FLOOR` | 7.5e-4 | Motion gating threshold |
| `SIM_THRESHOLD` | 0.58 | Similarity acceptance threshold |
| `SIM_MARGIN` | 0.10 | Top-1 vs Top-2 margin |
| `CONFIDENCE_TEMPERATURE` | 0.12 | Softmax temperature |
| `BEAM_SIZE` | 4 | Beam search width |
| `BEAM_LENGTH_PENALTY` | 0.8 | Length penalty alpha |
| `DTW_ENABLED` | true | Enable DTW re-ranking |
| `DTW_RADIUS` | 6 | DTW band constraint |
| `DTW_ALPHA` | 0.65 | Cosine/DTW blend weight |
| `DTW_LAMBDA` | 0.002 | DTW distance decay |
| `DTW_TOPK` | 5 | Candidates for DTW |
| `ENABLE_CUDA` | false | Allow GPU execution |

---

## 10. Troubleshooting

### Common Issues

1. **Empty translations**
   - Check: Tokenizer vocabulary matches training
   - Check: SOS/EOS token IDs are correct
   - Verify: `decoder_max_len` is sufficient

2. **Low translation quality**
   - Check: Motion gating threshold
   - Check: Input normalization (global_stats.npz)
   - Try: Increase beam size, adjust length penalty

3. **High latency**
   - Try: Reduce beam size (4 → 2)
   - Try: Enable CUDA if GPU available
   - Check: ONNX Runtime optimization level

4. **Motion gating rejections**
   - User may be signing too slowly
   - Lower `MOTION_FLOOR` threshold
   - Ensure good lighting for landmark detection
