# Sign2Text Model Improvement Plan

## 1. Diagnosis of Current Performance

Based on the training logs provided, the model is suffering from **Embedding Space Collapse** and **Decoder Mode Collapse**.

### A. Embedding Collapse (High NegSim)
- **Symptoms:** `PosSim` is ~0.96 (good), but `NegSim` is ~0.906 (very bad).
- **Meaning:** The model maps *all* videos to a very narrow cone in the vector space. A random negative sample is almost as similar to the anchor as the positive sample (margin ~0.06).
- **Consequence:** Retrieval fails (`Ret@1` ~5%) because the "nearest neighbor" search is essentially random within this narrow cluster.
- **Likely Causes:**
  - **Batch Size:** Contrastive learning (SupCon) requires large batch sizes to provide enough "hard negatives." A batch size of 8 or 16 is insufficient.
  - **Projection Head:** The features might be too raw.
  - **Lack of Text Alignment:** The encoder is learning to be invariant to augmentations (SupCon) but isn't being forced to align with the *semantics* of the text explicitly in the embedding space.

### B. Decoder Failure (WER ~1.0)
- **Symptoms:** The decoder generates fluent but irrelevant sentences (e.g., "can you tell me where the pain is" vs "may i have your health insurance card").
- **Meaning:** The decoder is ignoring the visual context (Posterior Collapse) and acting like a generic language model that just outputs common phrases from the training set.
- **Likely Causes:**
  - **Tokenizer:** The `SimpleTokenizer` (word-level) is likely sparse and brittle for an 11k dataset.
  - **Weak Encoder Signals:** Because of the embedding collapse, the encoder passes generic/similar vectors to the decoder for every input, so the decoder just guesses the most common sentences.

---

## 2. Strategic Improvements

### Step 1: Upgrade the Tokenizer & Text Model
Move away from the scratch-built `SimpleTokenizer`.
- **Action:** Use a pretrained Subword Tokenizer (e.g., `distilbert-base-uncased`).
- **Benefit:** Handles rare words better, reduces vocabulary size, and leverages transfer learning if we use the associated embeddings.
- **Implementation:**
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  vocab_size = tokenizer.vocab_size
  ```

### Step 2: Introduce Visual-Text Contrastive Loss (CLIP-style)
Currently, you have `SupCon` (Visual-Visual) and `CE` (Visual->Text). You are missing the direct link that forces the visual embedding to match the text embedding.
- **Action:** Add a **CLIP Loss** term.
- **Mechanism:**
  1. Encode Video $\rightarrow$ $V_{emb}$
  2. Encode Ground Truth Text (using a frozen or learnable text encoder, e.g., DistilBERT) $\rightarrow$ $T_{emb}$
  3. Maximize cosine similarity between correct $(V, T)$ pairs and minimize others.
- **Benefit:** This directly optimizes the `Ret@1` metric and forces the visual encoder to learn semantic features, which helps the decoder.

### Step 3: Fix the Embedding Geometry
- **Increase Batch Size:** This is critical for contrastive loss. If GPU memory is tight, use **Gradient Accumulation** (for CE loss) or a **Memory Bank / MoCo** approach (for Contrastive loss). For standard SupCon, try to push batch size to at least 64 or 128 using Mixed Precision (`fp16`).
- **Lower Temperature:** Try reducing the SupCon temperature from `0.07` to `0.05` to force the model to be sharper.
- **Separate Projection Heads:** Ensure the embedding used for retrieval/contrastive loss comes from a dedicated MLP projection head (e.g., `Linear -> ReLU -> Linear`) that is *not* used by the decoder. The decoder should consume the raw encoder output.

### Step 4: Model Scaling
For 11k videos (vs 1.5k), the model capacity should be increased slightly, but architecture depth is more important than width.
- **Embed Dim:** Increase `embed_dim` from 256 to **512**.
- **Encoder Layers:** Keep at 4-6.
- **Dropout:** Increase `dropout` slightly (0.2) to prevent overfitting to the specific motion patterns of the 11k set.

---

## 3. Recommended Configuration Changes

Update your `CFG` in the training script:

```python
CFG = {
    # ...
    "batch_size": 32,       # Try to maximize this (use gradient checkpointing if needed)
    "embed_dim": 512,       # Increase capacity
    "proj_dim": 256,        # Projection head output
    "temperature": 0.05,    # Sharpen contrastive loss
    "lr": 3e-4,             # Good starting point
    "weight_decay": 1e-4,   # Stronger regularization
    "tokenizer_model": "distilbert-base-uncased", # Use pretrained
    "loss_weights": {
        "supcon": 1.0,      # Visual-Visual alignment
        "clip": 1.0,        # Visual-Text alignment (NEW)
        "ce": 2.0           # Generation (increase weight)
    }
}
```

## 4. Next Steps for Implementation

1.  **Modify `SignDataset`:** Update `__getitem__` to return `input_ids` and `attention_mask` from the HuggingFace tokenizer.
2.  **Update Model:**
    - Replace `SimpleTokenizer` logic.
    - Add a `TextEncoder` module (can be `AutoModel.from_pretrained("distilbert-base-uncased")`) to generate text embeddings for the CLIP loss.
3.  **Update Training Loop:**
    - Calculate `loss_clip = contrastive_loss(visual_emb, text_emb)`.
    - `total_loss = loss_supcon + loss_clip + loss_ce`.

This approach directly targets the "High NegSim" (by improving contrastive setup) and "Bad Decoder" (by grounding the encoder in text semantics via CLIP).
