
Below is a **comprehensive master list** (organized from *quick wins → advanced improvements*).


---

## 🧩 **I. Core Inference Stabilization**

### 1️⃣ **Motion Energy Gating**

* Compute frame-to-frame Euclidean differences in landmark positions.
* If average motion < threshold (e.g. `1e-3`), skip inference or return `[still]`.
* Prevents idle frames from triggering predictions.

### 2️⃣ **Prediction Rejection / Thresholding**

* Analyze cosine similarity distributions (true vs. false matches).
* Set minimum acceptance rule, e.g.:

  * `max(similarity) ≥ 0.65` **AND**
  * `(top1 − top2) ≥ 0.10`
* If neither condition passes → output `"No clear match"`.

### 3️⃣ **Confidence Calibration**

* Replace raw `(sim + 1)/2` or `softmax(sim)` with temperature-scaled softmax:
  `confidence = softmax(similarities / τ)`
* Tune `τ` between `0.05–0.15` using validation embeddings.

### 4️⃣ **Temporal Prediction Smoothing**

* Maintain a queue (e.g. 5 recent predictions).
* Output a sentence only if ≥ 3 of last 5 match and motion_energy > threshold.
* Reduces jitter and flickering predictions.

---

## 🧠 **II. Preprocessing Alignment & Domain Fixes**

### 5️⃣ **Match Front-end and Training Preprocessing**

* Use the same:

  * frame sampling rate
  * normalization (mean/std, centering on torso)
  * sequence length (crop/pad to 64)
  * visibility weighting
* Inconsistent preprocessing between notebook and live app = domain drift.

### 6️⃣ **Visibility & Quality Weighting**

* Compute average `landmark.visibility` per frame.
* Drop low-visibility frames or down-weight them in sequence pooling.

### 7️⃣ **Camera Domain Adaptation**

* Augment training with webcam-like distortions:

  * horizontal flip (mirror)
  * random brightness / scaling / slight rotation
  * temporal jitter (drop random frames)
* Makes embeddings invariant to real camera setup.
(if there)

---

## ⚙️ **III. Representation & Training Improvements**

### 8️⃣ **Dynamic Time Warping (DTW) Alignment**

* Compare per-frame feature sequences using DTW or soft-DTW distance.
* Handles tempo mismatch and differing sign durations.
* Integrate as post-processing step before cosine similarity.

### 9️⃣ **Attention or Weighted Temporal Pooling**

* Replace mean pooling with attention weights or learnable pooling.
* Allows model to emphasize motion-rich frames and ignore idle segments.

### 🔟 **Add "No-Sign" / Idle Class**

* Insert idle/still samples labeled `[still]` or `[no_action]`.
* Train or fine-tune with a small classifier head to explicitly detect non-signing.

### 11️⃣ **Hard-Negative Mining**

* During training, ensure each batch includes visually similar but semantically different signs.
* Compute extra contrastive term to push these negatives apart.

### 12️⃣ **Supervised / Prototypical Fine-Tuning**

* Freeze encoder and:

  * Train a lightweight classifier (`Linear` or `MLP`) on sentence labels, or
  * Compute mean embedding per sentence and train with supervised contrastive loss.
* Produces globally organized embedding space instead of pairwise clusters.

### 13️⃣ **Embedding Normalization**

* Always L2-normalize encoder outputs before similarity comparison.
* Prevents scale variance from inflating confidence.

---

## 📊 **IV. Calibration & Evaluation Enhancements**

### 14️⃣ **Similarity Distribution Analysis**

* For validation data:

  * compute cosine sims for positive and negative pairs;
  * plot histograms;
  * find optimal operating threshold (max F1 or equal error rate).
* Guides confidence cutoff choice.

### 15️⃣ **Per-Category Recall Metrics**

* Track Recall@1/5 within each domain (e.g. Pediatrics, Pharmacy).
* Reveals imbalance or under-represented categories.

### 16️⃣ **Embedding Visualization**

* Use t-SNE/UMAP to visualize embeddings by sentence/category.
* Verify that clusters correspond to correct semantic groupings.

---

## 🧱 **V. Optional Architectural Upgrades**

### 17️⃣ **Use `model_complexity=2` in MediaPipe**

* Extract higher-fidelity landmarks for smoother sequences and cleaner inputs.
* No change needed in downstream model.

### 18️⃣ **Apply Temporal Denoising Filter**

* After extraction, smooth coordinates using a Gaussian or Savitzky–Golay filter.
* Removes jitter before feeding into encoder.

### 19️⃣ **Replace GRU with Temporal CNN or Transformer Encoder**

* If compute allows, swap Bi-GRU with 1D CNN + Self-Attention blocks.
* Improves temporal pattern recognition without heavy parameters.

### 20️⃣ **Mixed Precision Training**

* Enable `torch.cuda.amp.autocast()` for faster training and better stability on 4 GB GPU.

---

## 🧩 **VI. Real-World Calibration**

### 21️⃣ **Collect Small On-Device Calibration Set**

* Record 10–20 clips per sign on your actual webcam setup + negatives.
* Use to:

  * adjust similarity threshold,
  * fine-tune classifier,
  * test latency & stability.

### 22️⃣ **Prototype Nearest-Prototype Classifier**

* For each sentence, average multiple calibrated embeddings → prototype vector.
* During inference, compare live embeddings to prototypes instead of single clips.

---

## 🚀 **VII. Deployment Enhancements**

### 23️⃣ **FastAPI Inference API**

* Serve model with:

  * endpoint `/predict` accepting uploaded `.npy` or live frame stream,
  * motion gating + threshold logic built-in,
  * temperature-scaled confidences returned.

### 24️⃣ **Vector Database (Optional)**

* Store normalized embeddings in Postgres + pgvector / Supabase.
* Allows fast top-k similarity search and caching for scalability.

### 25️⃣ **Logging & Analytics**

* Log similarity scores, rejected samples, and latency to tune thresholds empirically.

---

## ✅ **Recommended Implementation Order**

1. **Add motion-energy gating & similarity threshold.**
2. **Align preprocessing (frontend ↔ training) & Domain Adaptation.**
3. **Gather calibration set & tune thresholds.**
4. **Add temporal smoothing + “no-sign” class.**
5. **Integrate DTW / attention pooling for temporal alignment.**
6. **(Optional)** supervised fine-tune or hard-negative mining.
7. **Deploy via FastAPI with calibrated inference.**

