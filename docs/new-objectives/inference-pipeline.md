This document outlines the architecture for the **Real-Time Inference Engine**. It bridges the gap between your static training script and a live, interactive demo.

Give this document to your agent to build the `signtalk_inference_demo.ipynb` notebook.

***

# Technical Spec: Real-Time Sign-to-Text Inference Engine
**Version:** 1.0
**Environment:** Jupyter Notebook / Local Python Environment
**Dependencies:** `opencv-python`, `mediapipe`, `torch`, `numpy`

---

## 1. System Overview
The inference engine is a **State Machine** that processes a continuous stream of frames (from a webcam or a simulated video file). It does not predict on every frame. Instead, it monitors "Visual Energy" to detect when a user starts and stops signing, creating dynamic "Proposed Windows" sent to the model.

### The Pipeline
1.  **Input Stream:** (Webcam or Concatenated Video)
2.  **Pose Extraction:** MediaPipe (Holistic/Pose).
3.  **Motion Gating:** Detects Start/Stop of gesture.
4.  **Buffer Management:** Accumulates frames during the "Active" state.
5.  **Normalization & Inference:** Prepares data $\rightarrow$ Model $\rightarrow$ Text.
6.  **Visualization:** Overlays skeleton and decoded text on the video feed.

---

## 2. Component Specifications

### 2.1. The `StreamBuffer` Class
Manages the sliding window of data.
*   **Attributes:**
    *   `buffer`: A `collections.deque` with maxlen (e.g., 100 frames).
    *   `state`: Enum [`IDLE`, `CAPTURING`, `COOLDOWN`].
    *   `last_prediction`: String (to display on screen).
*   **Methods:**
    *   `add_frame(landmarks)`: Pushes normalized coordinates.
    *   `clear()`: Empties buffer.
    *   `get_tensor()`: Returns `(1, T, 540)` tensor for the model.

### 2.2. The `MotionGate` (The Trigger)
Decides *when* to run the model to save compute and define sentence boundaries.
*   **Logic:** Tracks the velocity of **Wrist Keypoints**.
*   **Thresholds:**
    *   `START_THRESH`: Hand velocity > $0.05$ (Start Capturing).
    *   `STOP_THRESH`: Hand velocity < $0.02$ for $15$ consecutive frames (Stop & Predict).

### 2.3. The `VideoSim` Class (Synthetic Streamer)
To test robustness without sitting in front of a camera all day.
*   **Input:** A list of validation video paths `[v1.mp4, v2.mp4, v3.mp4]`.
*   **Process:**
    1.  Load Video 1.
    2.  Appends `30` frames of "black screen/zero pose" (simulating a user pausing).
    3.  Load Video 2.
    4.  Appends pause.
    5.  ...
*   **Output:** A Python generator that yields `frame` images one by one, mimicking a `cv2.VideoCapture` object. The concatenated videos are then stored in a location where they can be easily deleted

---

## 3. The Visualization Layer (UI)
The visual output must be rich for debugging.
*   **Skeleton Overlay:** Draw the MediaPipe landmarks on the frame.
    *   *Color Logic:*
        *   **Grey Skeleton:** IDLE state.
        *   **Red Skeleton:** RECORDING state (User is signing).
        *   **Green Skeleton:** SUCCESS state (Prediction returned).
*   **HUD (Heads Up Display):**
    *   Top Left: System State text.
    *   Bottom Center: **The Predicted Sentence.** (Large Font, Background Box).
    *   Top Right: Confidence Score bar.

---

## 4. Implementation Plan (Notebook Structure)

The Agent should structure the Jupyter Notebook as follows:

### Cell 1: Setup & Imports
*   Load the trained PyTorch model (`Seq2Seq` or `Classifier`).
*   Load the `normalize_frame_landmarks` function (Critical: Must match training preprocessing exactly).
*   Initialize MediaPipe.

### Cell 2: The Inference Class Definition
```python
class SignInferenceEngine:
    def __init__(self, model, labels_map):
        self.model = model
        self.buffer = []
        self.state = "IDLE"
    
    def process_stream(self, frame_stream_generator):
        # Main Loop
        pass
```

### Cell 3: The "Merged Video" Test (The Simulation)
*   **Goal:** Prove the model handles transitions.
*   **Logic:**
    1.  Select 3 random videos from the held-out validation set (e.g., "Hello", "Food", "House").
    2.  Stitch them into one long sequence.
    3.  Run the loop.
    4.  **Success Criteria:** The UI should update the text 3 times, corresponding to the 3 segments, without crashing during the "silence" in between.

### Cell 4: The Live Webcam Demo
*   Standard `cv2.VideoCapture(0)` loop.
*   Includes a `try/except` block to ensure the camera releases properly if the kernel is interrupted.

---

## 5. Agent Instructions (Prompt)

Copy/Paste this to your agent:

> **Task:** Create a Jupyter Notebook `signtalk_inference_demo.ipynb` for real-time testing.
>
> **1. Helper Functions:**
> *   Include the exact `normalize_frame_landmarks` function from the preprocessing script.
> *   Implement a `draw_styled_landmarks` function using OpenCV. It should change the connection color based on a `state` argument ('recording', 'idle').
>
> **2. Simulation Logic:**
> *   Create a generator function `simulate_video_stream(video_paths)` that opens videos sequentially, yields their frames, and yields 30 black frames between them.
>
> **3. The Inference Loop:**
> *   Process every frame through MediaPipe.
> *   **Gate Logic:** Calculate the Euclidean distance of Right/Left Wrists between current and previous frame.
> *   If `motion > threshold`: Set State = RECORDING, append keypoints to buffer.
> *   If `motion < threshold` AND `State == RECORDING` AND `buffer_len > 10`:
>     *   Wait 15 frames (debounce).
>     *   If still no motion: Set State = PREDICTING.
>     *   **Run Model:** Pass buffer to model -> Get Text.
>     *   Clear Buffer.
>     *   Set State = IDLE.
>
> **4. Visualization:**
> *   Use `cv2.putText` to display the `current_state` and `predicted_text` on the video frame.
> *   Use `cv2.imshow` (popped out window) for performance.
>
> **5. Test Case:**
> *   Load 3 distinct validation videos.
> *   Run the simulation loop and record the results.

---

### Why this visual approach is best
By simulating the stream first (Cell 3), you can debug the "Start/Stop" logic without waving your hands at the camera for hours. Once the "Merged Video" test passes (i.e., the model correctly identifies 3 separate sentences from one stream), you know the Webcam demo (Cell 4) will work perfectly.