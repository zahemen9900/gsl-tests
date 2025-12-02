# `signtalk_inference_demo.ipynb` Overview

This notebook walks through every step required to run offline and streaming inference for the GSL sign-to-text model: feature loading, normalization, tokenizer/model wiring, greedy decoding, stream simulation, visualization, and an optional webcam stub.

## Notebook Sections

1. **Environment + Config (Cells 1-2)** – Imports core libraries, defines inference hyperparameters, and loads dataset-wide normalization stats.
2. **Tokenizer + Model (Cells 3-6)** – Recreates the training-time tokenizer plus the projector/encoder/transformer decoder stack and greedy decode utilities with WER/BLEU metrics.
3. **Checkpoint + Batch Decoder (Cells 7-8)** – Loads the best checkpoint from `runs/`, preprocesses a sampled feature clip from `proc/processed_metadata.csv`, and prints a reference/prediction pair with metrics.
4. **Streaming Toolkit (Cells 9-12)** – Introduces `RollingFeatureBuffer`, `MotionEnergyGate`, stream simulators, and `run_streaming_inference`, then replays the sampled clip as if it were arriving in packets to observe partial hypotheses.
5. **Diagnostics + Visualization (Cells 13-14)** – Plots the motion-energy trace to tune gating thresholds and reason about latency vs. stability.
6. **Live Webcam Stub (Cells 15-17)** – Provides a guarded OpenCV/MediaPipe entry point for future real-time tests; currently fills pose vectors with placeholders so the flow can be integrated incrementally.

## Code Highlights

### Config + Stats Loading

```python
INFERENCE_CFG = {
    'seq_len': 64,
    'input_dim': 540,
    'proj_dim': 160,
    'rnn_hidden': 160,
    'rnn_layers': 2,
    'attn_heads': 4,
    'embed_dim': 256,
    'decoder_max_len': 64
}
GLOBAL_MEAN, GLOBAL_STD = load_global_stats(GLOBAL_STATS_PATH)
```

### Greedy Decode With Metrics

```python
@torch.no_grad()
def greedy_decode_batch(model, src_batch, tokenizer, max_len=64):
    proj = model.projector(src_batch)
    _, _, context = model.encoder(proj, return_sequence=True)
    # ... standard Transformer decoding loop ...
    return ys[:, 1:]

result = run_decoder_inference(seq2seq_model, sample_row['feature_path'], reference_text)
print('WER:', result['metrics']['wer'])
```

### Streaming Buffer + Gate

```python
buffer = RollingFeatureBuffer(feature_dim=INFERENCE_CFG['input_dim'], max_len=INFERENCE_CFG['seq_len'])
gate = MotionEnergyGate(window=emit_every, threshold=gate_threshold)
seq = buffer.append(frame)
if gate.update(seq[-chunk_hop:]):
    decoded = greedy_decode_batch(model, batch, tokenizer)
```

### Streaming Demo Invocation

```python
stream_packets = simulate_stream(sample_row, chunk_size=4)
stream_result = run_streaming_inference(
    seq2seq_model,
    stream_packets,
    tokenizer,
    chunk_hop=2,
    emit_every=4,
    gate_threshold=0.1
)
print('Streaming final transcript:', stream_result['final_transcript'])
```

### Webcam Stub Entry Point

```python
try:
    import cv2
except Exception as exc:
    cv2 = None

def decode_live_camera(model, tokenizer):
    if cv2 is None:
        print('cv2 missing; skipping live decode.')
        return
    seq = capture_webcam_stream()
    decoded = greedy_decode_batch(model, batch, tokenizer)
    print('Live prediction:', ids_to_sentence(decoded[0], tokenizer))
```

Use this summary to orient new contributors and reference the relevant cells quickly when tweaking inference behavior or documentation.
