# GhSL Encoder Architecture & Web Integration Guide

This document summarizes the end-to-end flow of the GhSL retrieval model and explains how to integrate the backend API into a browser experience (mirroring the implementation in `frontend/`).

---

## 1. System Overview

| Layer | Responsibilities | Key Artifacts |
| --- | --- | --- |
| **Client** | Capture webcam frames, extract MediaPipe landmarks, call inference API, render feedback | `frontend/src/App.tsx`, `frontend/src/services/api.ts`, `frontend/src/utils/features.ts` |
| **Backend API** | Normalize sequences, run ONNX encoder, apply retrieval logic, enforce gating, serve auxiliary metadata | `backend/app/services/inference.py`, `backend/app/main.py`, `backend/app/schemas.py` |
| **Model Assets** | Encoder graph, reference embeddings, prototypes, normalization stats, config snapshot | `backend/models/` (`sign_encoder.onnx`, `embeddings.npy`, `prototypes.npz`, `global_stats.npz`, `model_config.json`) |
| **Training Pipeline** | Generates artifacts and statistics consumed by the backend | `training.ipynb` (Cells 16–20), `proc/processed_metadata.csv` |

The flow is:

1. Browser records 64-frame sequences and extracts 540-dim features per frame via MediaPipe Holistic.
2. Browser submits the feature matrix (`frames`) to `POST /v1/inference`.
3. Backend normalizes frames with global stats, pads/trims to 64, runs the ONNX encoder, and compares against reference embeddings.
4. Backend responds with ranked predictions, gating metadata, and optional diagnostics (prototype similarity, DTW score).
5. Browser visualizes top predictions, latency, and optional overlay of captured landmarks over the recorded clip.

---

## 2. Model Architecture Flow

### 2.1 Preprocessing

1. **Landmark Extraction (client-side):** MediaPipe Holistic outputs pose/hand/face landmarks per frame. `frontend/src/utils/features.ts` converts landmarks to fixed-length 540-d feature vectors, applying torso normalization and down-sampling.
2. **Motion Gating (backend):** `compute_motion_energy` in `backend/app/services/inference.py` measures motion. If average motion `< motion_floor` (default `7.5e-4`), the request is rejected.
3. **Temporal Normalization:** `_prepare_sequence` crops longer clips to 64 frames (center crop) and pads shorter clips by repeating the last frame until 64 frames.
4. **Statistical Normalization:** Global mean/std from `global_stats.npz` are applied per feature dimension before inference.

### 2.2 Encoder

- **Frame Projector:** LayerNorm → Linear → GELU → Dropout → Linear → GELU reduces 540-d features to 160-d.
- **Temporal Encoder:** Bi-GRU (2 layers, hidden size 160) with motion-aware attention heads produces 256-d L2-normalized embeddings.
- **Training Objective:** Contrastive InfoNCE on augmented pairs to align sequences depicting the same sentence.

### 2.3 Retrieval & Decision Logic (backend)

1. **Initial Similarity:** Cosine similarity against `embeddings.npy` (shape `[num_sentences, 256]`).
2. **Optional DTW Re-ranking:** If `fastdtw` is installed and `DTW_ENABLED=true`, top-K entries are refined using motion-aware DTW distance.
3. **Prototype Agreement:** If `prototypes.npz` is available, the top-1 candidate must exceed `sim_threshold - 0.05` against the sentence prototype.
4. **Gating:** All of the following must pass for acceptance:
   - `similarity >= sim_threshold`
   - `(top1 - top2) >= sim_margin` (if second candidate exists)
   - Prototype check (if prototypes provided)
5. **Response Assembly:** Returns ranked predictions with metadata (confidence, motion mean, DTW diagnostics, acceptance flag).

---

## 3. Backend Integration Points

### 3.1 Configuration Inputs

Environment variables (see `backend/app/config.py`):

- `MODELS_DIR` (default `backend/models`)
- `ONNX_MODEL_PATH`, `EMBEDDINGS_PATH`, `MAPPING_PATH`
- `PROTOTYPES_PATH` (optional)
- `GLOBAL_STATS_PATH`
- `SEQ_LEN` (default `64`), `FEATURE_DIM` (default `540`)
- Gating thresholds: `MOTION_FLOOR`, `SIM_THRESHOLD`, `SIM_MARGIN`, `CONFIDENCE_TEMPERATURE`
- DTW settings: `DTW_ENABLED`, `DTW_RADIUS`, `DTW_ALPHA`, `DTW_LAMBDA`, `DTW_TOPK`
- Execution provider toggle: `ENABLE_CUDA` (defaults to `false`)
- Reference videos: `VIDEOS_DIR` (local) or `VIDEOS_BASE_URL` (remote storage, e.g., Cloudflare R2)

### 3.2 Public API Endpoints (FastAPI)

| Endpoint | Method | Purpose | Handler |
| --- | --- | --- | --- |
| `/healthz` | GET | Provider readiness check | `health_check` |
| `/v1/config` | GET | Returns encoder config for front-end tuning | `get_config` |
| `/v1/samples/random` | GET | Random sample metadata (sentence, category, optional video URL) | `random_sample` |
| `/videos/{filename}` | GET | Streams reference clip (local or redirect) | `stream_reference_video` |
| `/v1/inference` | POST | Main inference call | `run_inference` |

### 3.3 Request / Response Contract

Request (`SequencePayload`):
```json
{
  "frames": [[0.01, ... 540 floats ...], ...],
  "top_k": 5,
  "metadata": {"fps": 30, "source": "webcam"}
}
```

Response (`InferenceResponse`):
```json
{
  "top_k": 3,
  "results": [
    {
      "sentence_id": 42,
      "sentence": "THANK YOU",
      "category": "greeting",
      "video_file": "thank_you.mp4",
      "similarity": 0.83,
      "confidence": 0.91,
      "raw_similarity": 0.80,
      "proto_similarity": 0.78,
      "motion_mean": 0.0013,
      "accepted": true,
      "rejection_reason": null,
      "dtw_distance": 12.4,
      "dtw_score": 0.94
    }
  ]
}
```

Error responses surface as 400 (validation failures), 404 (missing media), or 503 (backend load failures).

---

## 4. Browser Integration Blueprint

The reference implementation (`frontend/src/App.tsx`) follows these steps:

1. **Bootstrap:** Fetch `/v1/config` to populate sequence length, feature dimension, and display the provider name.
2. **Get Sample:** Call `/v1/samples/random` to retrieve a sentence and reference video URL, then display the clip for the learner.
3. **Capture Loop Setup:**
   - Request webcam access (`navigator.mediaDevices.getUserMedia`).
   - Initialize MediaPipe Holistic with CDN-hosted models.
   - Start a manual `requestAnimationFrame` loop feeding frames to Holistic; register `onResults` callback.
4. **Feature Extraction:** In `onResults`, call `extractFrameData` to build:
   - `feature` – the 540-d float vector to append to `recordingBuffer`.
   - `landmarks` – cached for overlay visualization.
5. **Recording Session:**
   - Run a countdown, capture for `RECORDING_DURATION_SECONDS` (default 5s) into `recordingBuffer`.
   - Record optional WebM clip for playback.
6. **Inference Call:** Submit `recordingBuffer` to `runInference(frames, REQUIRED_TOP_K)` (wrapper around `POST /v1/inference`).
   - Measure latency via `performance.now()`.
   - Transition UI to success/failure based on whether target sentence appears in top-K list.
7. **Results Rendering:**
   - Show ranked predictions with confidence.
   - Visualize recorded clip with optional landmark overlay (`showLandmarks` toggle uses cached frames with colors for pose/hands/face).
   - Provide retry controls and allow picking another sample via `/v1/samples/random`.

### 4.1 API Helper (`frontend/src/services/api.ts`)

- `fetchConfig()` → GET `/v1/config`
- `fetchRandomSample()` → GET `/v1/samples/random`
- `runInference(frames, topK)` → POST `/v1/inference`

Ensure `API_BASE_URL` points to the deployed backend (set via `.env` or build-time config).

### 4.2 Feature Utility (`frontend/src/utils/features.ts`)

- Converts Holistic results to normalized coordinates.
- Enforces consistent ordering (pose → hands → face) matching training pipeline.
- Accesses canvas dimensions to scale landmarks to pixel coordinates for overlays.

### 4.3 UX States (App.tsx)

Stages: `landing → watching → countdown → recording → evaluating → success/failure`. Each state transitions based on timers, API promises, and gating outcomes.

---

## 5. Integration Checklist

1. **Backend:** Deploy FastAPI service with artifacts in `backend/models/` and environment variables set (see Section 3.1).
2. **CDN Access:** Ensure MediaPipe Holistic assets load via CDN (`https://cdn.jsdelivr.net/npm/@mediapipe/holistic/`).
3. **CORS:** `backend/app/main.py` allows `allow_origins=["*"]` by default. Restrict origins for production.
4. **HTTPS:** Host backend behind HTTPS; browsers block camera capture on insecure origins.
5. **Environment Config:** Provide `VITE_API_BASE_URL` in `frontend/.env` to point to the backend.
6. **Testing:**
   - Run backend locally: `uvicorn backend.app.main:app --reload`.
   - Start frontend dev server: `npm install && npm run dev` inside `frontend/`.
   - Smoke-test `/v1/inference` with exported `.npy` sequences if camera unavailable.
7. **Latency Monitoring:** Log round-trip latency from browser (`performance.now`) and backend metrics.

---

## 6. Extensibility Notes

- **Alternate Clients:** Mobile apps can reuse the same API; replace MediaPipe Holistic with platform-specific landmark extractors but maintain feature layout.
- **Model Updates:** After retraining, regenerate `embeddings.npy`, `prototypes.npz`, `global_stats.npz`, and `model_config.json`, then redeploy.
- **Batch Evaluation:** For offline datasets, call the backend with precomputed landmark arrays or invoke the ONNX model directly using `real_time_inference.py`.
- **Security:**
  - Validate request payload sizes server-side (e.g., limit frames ≤ 128).
  - Add authentication if exposing inference API publicly.

---

With this blueprint, you can integrate the GhSL encoder into other web experiences or reuse the backend for additional clients while keeping the preprocessing and retrieval assumptions consistent with the training pipeline.

