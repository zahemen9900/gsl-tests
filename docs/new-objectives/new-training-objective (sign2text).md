
# Technical Spec: Transitioning to Real-Time GhSL Translation
**Version:** 1.0
**Target:** Real-Time Pose-Based Sign Language Translation (Sign2Text)
**Current State:** Self-Supervised Contrastive Encoder (InfoNCE)
**Goal:** Sequence-to-Sequence Translation with Semantic Awareness

---

## 1. Executive Summary
The current model achieves a robust **Instance Discrimination (Inst@1: ~0.75)** but suffers from high **Negative Similarity (~0.80)**. This indicates the embedding space is clustered by "motion characteristics" rather than "semantic meaning."

To support a real-time 3D avatar and translation system, we must pivot from **Retrieval** (finding a similar video) to **Translation** (predicting text tokens from a stream). The architecture will evolve from a pure Encoder to an **Encoder-Decoder** or **CTC-based** system, utilizing **Supervised Contrastive Learning** to fix the embedding space before training the translation head.

---

## 2. Architectural Shift

### Current Architecture (Retrieval)
*   **Input:** Sequence of Pose Vectors $(T, 540)$.
*   **Backbone:** Transformer/GRU Encoder.
*   **Bottleneck:** Global Mean Pooling (collapses time).
*   **Objective:** `InfoNCE Loss` (Video A $\approx$ Augmented Video A).
*   **Inference:** Requires a database lookup (Vector Search). **High Latency.**

### Target Architecture (Translation)
*   **Input:** Sequence of Pose Vectors $(T, 540)$.
*   **Backbone:** Transformer Encoder (Preserves Time sequence).
*   **Head (New):** Auto-Regressive Transformer Decoder.
*   **Objective:** `CrossEntropy` (Token Prediction) + `SupCon` (Feature Alignment).
*   **Inference:** Direct token generation. **Low Latency.**

---

## 3. Data Strategy & Augmentation

### A. The "Many-to-One" Text Expansion
To handle semantic nuances, we will augment the **Text Labels**, not just the visual inputs.
*   **Logic:** A single sign sequence (Video A) often maps to multiple valid English translations.
*   **Implementation:**
    *   *Source:* Video A (Sign: "I eat apple").
    *   *Target 1:* "I am eating an apple."
    *   *Target 2:* "I consume fruit."
*   **Benefit:** This forces the Decoder to learn that the *visual representation* is flexible and corresponds to a semantic concept, preventing overfitting to rigid phrasing.

### B. Visual Augmentation (Pose-Level)
Continue using the existing pipeline (Random Rotate, Scale, Frame Drop) but ensure **Consistency Regularization**:
*   If Video A is augmented to Video A', the model should predict the *exact same text tokens* for both.

---


#### 3.1. The "Dual-View" Generation Pipeline
**Goal:** Generate positive pairs for Supervised Contrastive Learning without new recordings.
**Instruction:** Modify the `SignDataset` class to load a single `.npy` file and split it on-the-fly into two views:

*   **View 1 (Anchor):**
    *   Take frames at indices `0::2` (Even indices).
    *   Apply standard noise ($\sigma=0.01$).
*   **View 2 (Positive):**
    *   Take frames at indices `1::2` (Odd indices).
    *   **Apply Y-Axis Rotation:** Rotate all $(x,y,z)$ coordinates by $\theta \in [-15^{\circ}, 15^{\circ}]$.
    *   *Math:* $x' = x \cos(\theta) + z \sin(\theta)$; $z' = -x \sin(\theta) + z \cos(\theta)$.

**Text Label Handling:**
*   Use the `google/genai` API to generate 5 variations of the sentence (e.g., "I go home", "Going home", "I am heading home").
*   In the `__getitem__` method:
    *   Assign **Text Label A** (randomly chosen from the 5) to **View 1**.
    *   Assign **Text Label B** (different random choice) to **View 2**.
*   **Effect:** The model learns that *Visually Different Inputs* (View 1 vs 2) = *Semantically Identical Meanings*.

---


## 4. Implementation Roadmap

### Phase 1: Feature Space Cleanup (Supervised Contrastive)
*Goal: Lower the Negative Similarity (0.80) and group semantically identical videos.*

1.  **Modify DataSampler:** Instead of random batches, implement a `ClassBalancedSampler`. Ensure every batch contains at least 2 distinct videos of the *same sign/sentence*.
2.  **Change Loss Function:** Switch from `InfoNCE` (Self-Supervised) to **`SupConLoss` (Supervised Contrastive Loss)**.
    *   *Logic:* Pull embeddings of "Video A (Hello)" and "Video B (Hello)" together; Push "Video C (Goodbye)" away.
3.  **Metric to Watch:** `NegSim` must drop below **0.5**. `PosSim` should remain > 0.8.

### Phase 2: The Translation Head (Seq2Seq)
*Goal: Convert pose sequences to text.*

1.  **Remove Pooling:** Disable the `mean(dim=1)` operation in the Encoder. The output must be $(B, T, Hidden\_Dim)$.
2.  **Attach Decoder:** Add a Transformer Decoder (2-4 layers).
    *   *Cross-Attention:* Queries from Text, Keys/Values from Visual Encoder output.
3.  **Tokenization:** Use a Subword Tokenizer (BPE) or a Gloss-level vocabulary on your target text.
4.  **Training Objective:** Standard `CrossEntropyLoss` with **Teacher Forcing**.

### Phase 3: Real-Time Inference Strategy
1.  **Sliding Window:** Implement a buffer (e.g., 50 frames) with a stride (e.g., 10 frames).
2.  **End-of-Sentence Token:** The model must learn to predict `<EOS>`. When `<EOS>` is detected, flush the buffer and display the sentence.

---

## 5. Metrics for Success

Do not rely on Loss or Contrastive Accuracy anymore. Use Generative Metrics.

| Metric | Description | Target Goal (PoC) |
| :--- | :--- | :--- |
| **WER (Word Error Rate)** | (Substitutions + Deletions + Insertions) / Total Words. Lower is better. | **< 30%** |
| **BLEU-4** | Measures n-gram overlap between prediction and reference translations. Higher is better. | **> 20.0** |
| **NegSim (Encoder)** | Similarity of unrelated signs. | **< 0.5** |
| **Inference Time** | Time from frame input to text output. | **< 100ms** |

---

## 6. Direct Instructions for the Agent

**Prompt to Agent:**
> "Refactor the `signtalk_local_training.py` script.
> 1.  **Data Loader:** Update `SignDataset` to accept a list of synonymous sentences for each video. During training, randomly select one text variation per epoch for the label.
> 2.  **Encoder:** Keep the `TemporalEncoder` but add a flag `return_sequence=True` to bypass the mean pooling layer.
> 3.  **New Module:** Create a `SignTranslationModel` class that combines the Encoder with a `nn.TransformerDecoder`.
> 4.  **Loss:** Implement a dual-loss approach:
>     *   `Loss = λ * SupConLoss(encoder_features) + (1-λ) * CrossEntropy(decoder_text)`
> 5.  **Evaluation:** Replace the 'Retrieval' loop with a 'Generation' loop using greedy decoding to calculate WER."

---

### Why this approach works for you:
*   **Leverages your 9k data:** The `SupConLoss` makes the most of your limited data by learning tight clusters before asking the model to learn grammar.
*   **Synthetic Text:** Using multiple text variations prevents the model from memorizing "Video 42 = Sentence 42" and instead teaches "Video Features X = Semantic Concept Y."
*   **Real-Time:** The Seq2Seq model is computationally light enough for edge inference compared to video-pixel models.