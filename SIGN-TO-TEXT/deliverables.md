# Project Deliverables: Transition to Real-Time GhSL Translation

**Objective:** Transition from a Retrieval-based system (InfoNCE) to a Generative Translation system (Seq2Seq) for Real-Time Ghanaian Sign Language interpretation.

---

## 📅 Sprint Plan

### Sprint 1: Data Pipeline & Preprocessing Refactor
**Goal:** Enable "Dual-View" generation and "Many-to-One" text augmentation to support Supervised Contrastive Learning.

*   **Task 1.1: Text Augmentation Pipeline**
    *   [ ] Integrate `google/genai` or similar API to generate 3-5 synonymous sentences for each existing label in `processed_metadata.csv`.
    *   [ ] Update metadata schema to store list of synonyms (e.g., `text_variations` column).
*   **Task 1.2: Dual-View Data Loading**
    *   [ ] Refactor `SignDataset` in `training.ipynb` (or new script).
    *   [ ] Implement on-the-fly splitting of `.npy` files:
        *   **View 1 (Anchor):** Even frames + Noise.
        *   **View 2 (Positive):** Odd frames + Y-Axis Rotation.
    *   [ ] Update `__getitem__` to return `(view1, view2, text_label_1, text_label_2)`.
*   **Task 1.3: Class Balanced Sampler**
    *   [ ] Implement `ClassBalancedSampler` to ensure batches contain at least 2 distinct videos of the same sign/sentence (crucial for SupCon).

### Sprint 2: Model Architecture Transition
**Goal:** Build the Sequence-to-Sequence architecture.

*   **Task 2.1: Encoder Refactoring**
    *   [ ] Modify `TemporalEncoder` to accept a `return_sequence=True` flag.
    *   [ ] Remove Global Mean Pooling when this flag is active. Output shape: `(B, T, Hidden_Dim)`.
*   **Task 2.2: Decoder Implementation**
    *   [ ] Create `SignTranslationModel` class.
    *   [ ] Integrate `nn.TransformerDecoder` (2-4 layers).
    *   [ ] Implement Cross-Attention mechanism (Queries=Text, Keys/Values=Visual Encoder).
*   **Task 2.3: Tokenization**
    *   [ ] Select and implement a Tokenizer (BPE or Word-level) for the target text.
    *   [ ] Build vocabulary from the augmented text dataset.

### Sprint 3: Training Pipeline & Loss Integration
**Goal:** Implement the dual-loss training loop.

*   **Task 3.1: Supervised Contrastive Loss (SupCon)**
    *   [ ] Implement `SupConLoss` module.
    *   [ ] Replace `InfoNCE` with `SupConLoss` for the Encoder output.
*   **Task 3.2: Dual-Objective Training Loop**
    *   [ ] Implement the combined loss function:
        $$Loss = \lambda \cdot \mathcal{L}_{SupCon} + (1-\lambda) \cdot \mathcal{L}_{CrossEntropy}$$
    *   [ ] Implement **Teacher Forcing** for the Decoder training.
*   **Task 3.3: Training Script Overhaul**
    *   [ ] Create `train_seq2seq.py` (or update notebook).
    *   [ ] Add logging for both loss components separately.

### Sprint 4: Inference & Evaluation
**Goal:** Enable real-time text generation and proper metrics.

*   **Task 4.1: Decoding Strategies**
    *   [ ] Implement **Greedy Decoding** for fast inference.
    *   [ ] (Optional) Implement **Beam Search** for better quality.
*   **Task 4.2: Evaluation Metrics**
    *   [ ] Implement **WER (Word Error Rate)** calculation.
    *   [ ] Implement **BLEU-4** score calculation.
    *   [ ] Remove reliance on "Recall@k" metrics.
*   **Task 4.3: Real-Time Simulation**
    *   [ ] Implement a sliding window buffer (e.g., 50 frames, stride 10).
    *   [ ] Implement `<EOS>` token detection to trigger output.

---

## 📊 Success Metrics

| Metric | Target Goal (PoC) |
| :--- | :--- |
| **WER (Word Error Rate)** | **< 30%** |
| **BLEU-4** | **> 20.0** |
| **NegSim (Encoder)** | **< 0.5** |
| **Inference Time** | **< 100ms** |

---

## 🛠 Technical Requirements

*   **Libraries:** `torch`, `transformers`, `sacrebleu` (for BLEU), `jiwer` (for WER).
*   **Hardware:** Current RTX 3050 (4GB) requires strict memory management (Mixed Precision `fp16` is mandatory).
