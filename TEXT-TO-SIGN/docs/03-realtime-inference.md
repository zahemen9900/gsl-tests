# Text-to-Sign: Real-Time Inference Engineering

## 1. System Architecture

The inference pipeline is designed to run in a browser-based environment (Web/React) or a Game Engine (Unity) at **60 FPS** with minimal latency (<100ms trigger-to-render time).

### 1.1 The Pipeline
```mermaid
graph LR
    User[User Input] -->|Text| NLP[NLP Service]
    NLP -->|Gloss/Tokens| Inf["Inference Engine (ONNX)"]
    Inf -->|Raw Poses| Filter[Signal Filtering]
    Filter -->|Smooth Poses| IK[Hybrid IK Solver]
    IK -->|Rotations| Avatar[3D Avatar]
```

---

## 2. Text Pre-Processing (NLP Service)

Before the neural network sees the input, we must bridge the gap between English and GhSL.

### 2.1 Text Normalization
- **Lowercasing**: "Hello" $\rightarrow$ "hello"
- **Punctuation Stripping**: "How are you?" $\rightarrow$ "how are you"
- **Synonym Mapping**: Use a dictionary to map rare words to common glosses known by the model.
    - *Example*: "Physician" $\rightarrow$ "DOCTOR"
    - *Example*: "Medic" $\rightarrow$ "DOCTOR"

### 2.2 Gloss Conversion (LLM-Lite)
We use a small, distilled LLM (e.g., flan-t5-small or a fine-tuned GPT-2) to convert English grammar to Sign Grammar (Subject-Object-Verb, Topic-Comment).
- *Input*: "I am not going to the store."
- *Output*: `[STORE] [GO] [NOT]`

*Note*: For the MVP, we can skip this and train the main model on English text directly, relying on the `DistilBERT` adapter to learn the mapping, but an explicit Gloss step usually improves quality.

---

## 3. The Inference Engine (ONNX Runtime)

We export the PyTorch Generator to ONNX for cross-platform compatibility.

### 3.1 Model Export Config
- **Opset Version**: 17+ (Required for Transformer layers).
- **Dynamic Axes**:
    - Batch Size (Support batched requests).
    - Sequence Length (Output length varies).
- **Quantization**: FP16 (Half Precision) is recommended for WebGL/WebGPU backends to reduce download size (~50MB $\rightarrow$ ~25MB).

### 3.2 Streaming Generation (Chunking)
For long sentences, we do not wait for the entire sequence to generate. The **Non-Autoregressive** model generates the whole sentence at once, but if we implement a **sliding window** approach for paragraphs, we can stream:
1.  Generate Sentence 1.
2.  Start playing Sentence 1.
3.  Background generate Sentence 2.
4.  Blend Sentence 1 end $\rightarrow$ Sentence 2 start.

---

## 4. Signal Processing & Retargeting

Raw model output $(x, y, z)$ contains high-frequency noise and lacks physical constraints.

### 4.1 The 1€ Filter (Jitter Removal)
We use the **1€ Filter** (One Euro Filter), an adaptive low-pass filter.
- **Why?** Standard interaction requires *low latency* (for fast signs) and *high smoothing* (for static holds).
- **Configuration**:
    - `min_cutoff` (1.0 Hz): Stability when holding still.
    - `beta` (0.007): Sensitivity to speed. Higher beta = less lag, more jitter.

### 4.2 Hybrid IK (Retargeting)
We map the 540 topological points to the Avatar's bones.

**A. Arms (Analytical 2-Bone IK)**
- *Goal*: Solves Shoulder and Elbow rotations to place the Wrist at the target $(x,y,z)$.
- *Pole Vector*: Use the ML-predicted Elbow position to determine the "swivel" of the arm.

**B. Hands (Local Rotation FK)**
- *Goal*: Map finger landmarks to knuckle rotations.
- *Algorithm*:
    - For each finger joint, calculate vector $v = P_{child} - P_{parent}$.
    - Transform global $v$ to the parent bone's local space.
    - Convert vector direction to Quaternion rotation.
    - *Clamp*: Limit rotations to human range (e.g., fingers can't bend backwards).

**C. Body/Torso (Look-At FK)**
- The Spine and Head simply "Look At" their child joints or average targets, providing a natural sway.

---

## 5. Performance Targets

| Component | Target Budget (ms) | Notes |
|-----------|--------------------|-------|
| NLP / Tokenization | 10ms | Run on CPU / Worker |
| Model Inference | 30ms | Run on GPU (WebGPU/WebGL) |
| IK Solving | 2ms | Very cheap analytical math |
| Rendering | 12ms | Standard 3D rendering |
| **Total** | **~54ms** | Fits within 60 FPS (16ms per frame is render budget; inference runs async) |
