# Text-to-Sign: Data Preparation & Preprocessing Strategy

## 1. Overview & Data Reuse Strategy

We will leverage the existing high-quality dataset curation pipeline established in `signtalk_colab_sampling.md` but adapt the feature extraction phase to support **generative** tasks. Unlike classification (Sign2Text), which benefits from invariance (ignoring speed/style differences), generation (Text2Sign) requires **variance** preservation to produce natural, non-robotic motion.

### 1.1 Reuse of `SignTalk_Sampled`
We will reuse the **raw video sampling strategy** defined in `signtalk_colab_sampling.md` to ensure our training data is:
1.  **Balanced**: Stratified by category (Healthcare, etc.).
2.  **Diverse**: Includes multiple variants (Signers A, B, C...) for the same sentence ID.
3.  **Clean**: Free from corrupt or "unmapped" videos.

**Action Item:**
- Run the `signtalk_colab_sampling.ipynb` logic exactly as is to generate the `SignTalk-GH_Sampled.zip` containing ~1,200 - 2,400 raw videos.
- **Do not** apply the aggressive "low-motion rejection" from the classification pipeline yet; we need to analyze if "pauses" are semantic (e.g., holding a sign for emphasis).

---

## 2. Preprocessing Pipeline (Generative Adaptation)

The feature extraction pipeline (`signtalk_local_preprocessing.md`) needs specific modifications for *generation*.

### 2.1 Feature Extraction (MediaPipe Holistic)
We will continue using MediaPipe Holistic but with a stricter focus on **temporal consistency**.

*   **Landmarks**: 540 dimensions (Pose 33x4 + Hands 21x3x2 + Face 468/5x3).
*   **World Coordinates**: Use `pose_world_landmarks` (meters) instead of `pose_landmarks` (normalized screen, 0-1) where possible. World coordinates are critical for the **Geometric Losses** (bone length consistency) used in the GAN training.
    *   *Note*: MediaPipe's "World" coordinates are centered on the hips in meters. This saves us a normalization step but requires validation.

### 2.2 Normalization (The "Canonical Avatar" Space)
To train a single model on multiple signers, we must normalize their skeletons to a "Canonical Avatar" while preserving their *motion*.

**Step 1: Root Centering**
- For every frame $t$, calculate the Hip Center: $P_{root, t} = \frac{P_{left\_hip, t} + P_{right\_hip, t}}{2}$.
- Subtract $P_{root, t}$ from all joints.
- *Outcome*: The character always walks in place; the root is at $(0,0,0)$.

**Step 2: Skeletal Rescaling (Fixed Height)**
- Calculate the "Refrence Spine Length" of the signer (Distance from Mid-Hip to Mid-Shoulder).
- Scale the entire point cloud so that this Spine Length equals exactly **1.0 unit**.
- *Outcome*: A 150cm signer and a 190cm signer will have the same arm reach relative to their body.

**Step 3: Forward Facing Alignment**
- Calculate the "Shoulder Vector": $V_{shoulders} = P_{right\_shoulder} - P_{left\_shoulder}$.
- Rotate the point cloud around the Y-axis (up) so $V_{shoulders}$ is parallel to the X-axis.
- *Outcome*: The signer always faces forward ($+Z$).

### 2.3 Motion Smoothing & FPS Normalization
Generative models are highly sensitive to jitter.

1.  **Imputation**: Linearly interpolate missing frames (where `visibility < threshold`).
2.  **1€ Filter (Offline)**: Apply a light 1€ filter during preprocessing to clean "camera buzz" while keeping fast hand flicks (which low-pass filters destroy).
3.  **Fixed FPS**: Resample all clips to exactly **30 FPS**. Variable frame rates will confuse the Length Predictor.

---

## 3. Tokenization & Text Processing

For Text-to-Sign, the input is Text, not Video. We need a robust text processing pipeline.

### 3.1 Input Format
We will support two input modes:
1.  **Raw English**: "I am going to the hospital."
2.  **Gloss (Preferred)**: `[ME] [GO] [HOSPITAL]`

### 3.2 Text Embeddings
Instead of training a simple embedding layer from scratch (which requires massive data), we will use a **Pre-trained Language Model (frozen)** with a trainable adapter.

*   **Encoder**: `DistilBERT` or `MiniLM-L6-v2`.
*   **Mechanism**:
    1.  Tokenize input string.
    2.  Pass through DistilBERT $\rightarrow$ Get `[CLS]` token + Sequence Embeddings $(T_{text}, 384)$.
    3.  **Adapter Layer**: A small MLP projecting $384 \rightarrow 256$ (Model Dimension).

This allows the model to understand that "Physician" and "Doctor" are semantically identical without needing thousands of examples of both.

---

## 4. Final Dataset Artifacts

After preprocessing, the dataset directory should look like this:

```
dataset/
├── raw_videos/              # From Sampling Step
├── features/
│   ├── XYZ.npy              # (T, 540) - Normalized, Centered, World coords
│   └── XYZ_meta.json        # {"spine_scale": 1.2, "original_fps": 24}
├── metadata.csv
│   ├── video_id             # XYZ
│   ├── sentence_text        # "Where is the doctor?"
│   ├── sentence_gloss       # "[WHERE] [DOCTOR]" (Generated via LLM if missing)
│   ├── total_frames         # 64
│   └── signer_id            # "Signer_A"
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### 4.1 Data Augmentation (Pre-computed)
To handle the limited data (~4000 sentences), we will use the **Gemini API** again (as in `training.ipynb`) to pre-generate purely text-based augmentations:
- *Original*: "Where is the pharmacy?"
- *Aug 1*: "im looking for the pharmacy"
- *Aug 2*: "location of pharmacy please"

These variations map to the **same** video ID, teaching the model robust NLU.
