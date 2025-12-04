### Part 2: Technical Spec for Model B (Text-to-Sign)

This model operates in reverse. It is a **Generative Model**. It takes text and hallucinates human motion.

**The Challenge:**
If you train a standard model to predict motion, it often outputs "the average of all motions," resulting in an avatar that looks like it's floating or sliding (the "Zombie Effect"). This spec includes specific countermeasures for that.

***

# Technical Spec: Real-Time Text-to-Sign Generation (T2S)
**Version:** 1.0
**Goal:** Generate fluid 3D Skeletal Animation from English/Gloss Input.
**Target Output:** Drive a Unity/Web 3D Avatar (Rigged Humanoid).
**Data Source:** Inverted Dataset (Text $\rightarrow$ Normalized Pose Coordinates).

---

## 1. Architecture: The "Progressive Motion Transformer"

We will use a **Non-Autoregressive Transformer** (or a fast Autoregressive one) to ensure low latency.

### A. The Inputs
*   **Source:** Text Tokens (English or Gloss).
*   **Positional Encoding:** Standard Sinusoidal encoding.

### B. The Model Structure (Encoder-Decoder)
1.  **Text Encoder:**
    *   Takes text tokens.
    *   Outputs a "Context Vector" representing the semantic meaning.
    *   *Size:* Small (2 Layers, 256 dim).
2.  **Length Predictor (Crucial Component):**
    *   A small MLP head attached to the Encoder.
    *   *Task:* Look at the text embedding and predict **How many frames** this sign sequence should be.
    *   *Why:* You cannot generate motion if you don't know when to stop.
3.  **Motion Decoder:**
    *   **Input:** A sequence of "Seed Queries" (zeros or noise) of length $L$ (predicted by the Length Predictor).
    *   **Cross-Attention:** Attends to the Text Encoder outputs.
    *   **Output:** $(L, \text{Joints} \times 3)$ coordinate sequence.

---

## 2. The Loss Function (The "Anti-Zombie" Recipe)

Standard MSE (Mean Squared Error) is not enough. You need a **Composite Loss Function** to ensure the motion looks human.

$$ \mathcal{L}_{total} = \lambda_{pos}\mathcal{L}_{MSE} + \lambda_{vel}\mathcal{L}_{vel} + \lambda_{bone}\mathcal{L}_{bone} $$

1.  **$\mathcal{L}_{MSE}$ (Position Loss):**
    *   Ensures the hand is in the right place (e.g., near the head for "Think").
2.  **$\mathcal{L}_{vel}$ (Velocity/Smoothness Loss):**
    *   Calculates the difference between Frame $t$ and Frame $t-1$.
    *   Ensures the *speed* of the generated motion matches the *speed* of the real video. Prevents the avatar from "teleporting" hands.
3.  **$\mathcal{L}_{bone}$ (Limb Consistency Loss):**
    *   **Critical for Avatars.**
    *   The distance between the *Shoulder* and *Elbow* must be constant.
    *   If the model predicts a pose where the arm stretches 2 meters, this loss penalizes it heavily.

---

## 3. Data Pipeline & Augmentation

### 3.1. Text Normalization
*   **Input:** "I am going to the market."
*   **Process:**
    1.  Convert to Lowercase.
    2.  (Optional but Recommended) Convert to **Gloss** using an LLM API before feeding to the model.
        *   *LLM Prompt:* "Convert 'I am going to the market' to GhSL Gloss." $\rightarrow$ `[ME] [GO] [MARKET]`
    3.  Tokenize.

### 3.2. Skeleton Normalization (Retargeting Prep)
The model output will be $X, Y, Z$ coordinates.
*   **Issue:** MediaPipe coordinates are not rotations. Avatars usually need **Joint Rotations (Quaternions)**.
*   **Solution for PoC:**
    *   Train the model on **Positions** (easier to learn).
    *   Use an **Inverse Kinematics (IK)** solver in your frontend (Unity/React Three Fiber) to map these positions to the Avatar's bones.
    *   *Note:* This simplifies the ML model significantly.

---

## 4. Inference Pipeline (The "Real-Time" Flow)

When the user types "Hello":

1.  **Text Encoding:** Model creates context vector.
2.  **Duration Prediction:** Model predicts: "This takes 45 frames."
3.  **Generation:** Model outputs 45 frames of coordinates $(45, 540)$.
4.  **Post-Processing (Smoothing):**
    *   Raw model output might jitter.
    *   Apply a **Savitzky-Golay Filter** or simple **Gaussian Smoothing** over the time axis to make the motion buttery smooth.
5.  **Frontend Display:**
    *   Stream these points to the Avatar's IK targets (Left Hand Target, Right Hand Target, Head Target).

---

## 5. Agent Instructions (Prompt)

Copy-paste this to your AI agent to build the Training Script:

> **Task:** Create a `signtalk_text_to_pose.py` training script.
>
> **1. Architecture:**
> *   Build a Transformer-based Generator (`TextToPoseModel`).
> *   **Encoder:** `nn.TransformerEncoder` (Text -> Hidden).
> *   **Length Estimator:** `nn.Linear` (Hidden -> Scalar `num_frames`).
> *   **Decoder:** `nn.TransformerDecoder`. Input is a positional encoding sequence of length `num_frames`. Output is `(Batch, Frames, 540)`.
>
> **2. Data Loader:**
> *   Reverse the logic of the previous loader.
> *   Input: Text Token IDs.
> *   Target: Normalized Pose Coordinates `.npy`.
> *   *Crucial:* Pad all pose sequences to a fixed max length (e.g., 128) and use a `src_key_padding_mask` so the model ignores padding.
>
> **3. Loss Function:**
> *   Implement a custom `MotionLoss` class.
> *   It must calculate MSE on coordinates.
> *   It must calculate L1 loss on the *first derivative* (velocity) to encourage temporal smoothness.
>
> **4. Training Loop:**
> *   Forward pass: Predict Length $\rightarrow$ Create Mask $\rightarrow$ Predict Poses.
> *   Backward pass: Backpropagate through both Length loss (MSE against real length) and Motion loss.

---

### Why this works for your constraints
*   **IK Solver Strategy:** By offloading the "Bone Rotation" math to the game engine/frontend (via Inverse Kinematics), your ML model becomes much simpler (just predicting dots in space).
*   **Synthetic Data:** You mentioned generating synthetic sentences. This is **gold** here. You can train the model on 50 variations of English text mapping to the *same* video. This teaches the model that "Hi", "Hello", and "Greetings" all produce the *same* wave motion.