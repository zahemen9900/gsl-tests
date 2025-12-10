# GhSL Sign2Text: Backend API Service
**Version:** 1.0  
**Status:** ✅ Production-Ready  
**Last Updated:** December 2024

---

## Executive Summary

This document describes the **FastAPI Backend Service** that serves the GhSL Sign2Text translation model. The backend handles model loading, request processing, inference execution, and provides a RESTful API for frontend consumption. It supports both ONNX-based embedding retrieval and real-time streaming inference.

### Key Features

- **FastAPI Framework** with async request handling
- **ONNX Runtime** for optimized inference
- **Thread-safe** model access with locking
- **Motion gating** for quality control
- **DTW re-ranking** for improved retrieval accuracy
- **Prototype agreement** checks for confidence validation
- **CORS-enabled** for cross-origin frontend access
- **Health checks** for deployment monitoring

---

## 1. Service Architecture

### 1.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │    Routers   │  │  Middleware  │  │    Dependencies    │    │
│  │ - inference  │  │ - CORS       │  │ - get_service()    │    │
│  │ - metadata   │  │              │  │ - get_settings()   │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 OnnxSignInferenceService                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │   │
│  │  │ ONNX Session │  │  Embeddings  │  │  Prototypes  │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Model Artifacts                          │
│  sign_encoder.onnx │ embeddings.npy │ global_stats.npz │ ...    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 File Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── __main__.py          # Entry point: python -m backend.app
│   ├── main.py              # FastAPI app definition & routes
│   ├── config.py            # Settings from environment
│   ├── dependencies.py      # Dependency injection
│   ├── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   └── inference.py     # Real-time inference endpoints
│   └── services/
│       └── inference.py     # OnnxSignInferenceService
├── models/                  # Inference artifacts (symlink to runs/)
├── requirements.txt
├── run.sh
└── README.md
```

---

## 2. Configuration System

### 2.1 Settings Dataclass

```python
@dataclass(frozen=True)
class Settings:
    # Paths
    seq_len: int
    feature_dim: int
    top_k: int
    models_dir: Path
    onnx_model_path: Path
    embeddings_path: Path
    mapping_path: Path
    prototypes_path: Optional[Path]
    global_stats_path: Path
    
    # Runtime
    use_gpu: bool
    videos_dir: Optional[Path]
    videos_base_url: Optional[str]
    
    # Inference thresholds
    motion_floor: float
    sim_threshold: float
    sim_margin: float
    confidence_temperature: float
    
    # DTW settings
    dtw_enabled: bool
    dtw_radius: int
    dtw_alpha: float
    dtw_lambda: float
    dtw_topk: int
```

### 2.2 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `backend/models` | Directory for inference artifacts |
| `ONNX_MODEL_PATH` | `${MODELS_DIR}/sign_encoder.onnx` | ONNX model path |
| `EMBEDDINGS_PATH` | `${MODELS_DIR}/embeddings.npy` | Reference embeddings |
| `MAPPING_PATH` | `${MODELS_DIR}/embeddings_map.csv` | Metadata mapping |
| `PROTOTYPES_PATH` | `${MODELS_DIR}/prototypes.npz` | Optional prototypes |
| `GLOBAL_STATS_PATH` | Auto-detected | Normalization statistics |
| `SEQ_LEN` | `64` | Nominal sequence length |
| `FEATURE_DIM` | `540` | Feature dimension |
| `DEFAULT_TOP_K` | `5` | Default retrieval fan-out |
| `MOTION_FLOOR` | `7.5e-4` | Motion gating threshold |
| `SIM_THRESHOLD` | `0.58` | Acceptance similarity |
| `SIM_MARGIN` | `0.10` | Top-1 vs Top-2 margin |
| `CONFIDENCE_TEMPERATURE` | `0.12` | Softmax temperature |
| `ENABLE_CUDA` | `false` | Allow GPU providers |
| `DTW_ENABLED` | `true` | Enable DTW re-ranking |
| `DTW_RADIUS` | `6` | DTW band constraint |
| `DTW_ALPHA` | `0.65` | Cosine/DTW blend |
| `DTW_LAMBDA` | `0.002` | DTW distance decay |
| `DTW_TOPK` | `5` | DTW candidates |
| `VIDEOS_DIR` | Auto-detected | Reference video directory |
| `VIDEOS_BASE_URL` | None | Remote video CDN URL |

### 2.3 Settings Loading

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings.from_env()
```

---

## 3. API Endpoints

### 3.1 System Endpoints

#### Health Check

```http
GET /healthz
```

**Response:**
```json
{
  "status": "ok",
  "detail": "provider=CPUExecutionProvider"
}
```

#### Configuration

```http
GET /v1/config
```

**Response:**
```json
{
  "seq_len": 64,
  "feature_dim": 540,
  "default_top_k": 5,
  "model_provider": "CPUExecutionProvider",
  "motion_floor": 0.00075,
  "sim_threshold": 0.58,
  "sim_margin": 0.10,
  "confidence_temperature": 0.12,
  "dtw_enabled": true
}
```

### 3.2 Metadata Endpoints

#### Random Sample

```http
GET /v1/samples/random
```

**Response:**
```json
{
  "sentence_id": 42,
  "sentence": "THANK YOU",
  "category": "greeting",
  "video_file": "thank_you.mp4",
  "video_url": "/videos/thank_you.mp4"
}
```

#### Sample Batch (Legacy)

```http
GET /samples
```

**Response:**
```json
[
  {
    "video_file": "hello.mp4",
    "sentence": "HELLO",
    "feature_path": "features/pose_data/hello.npy"
  }
]
```

### 3.3 Media Endpoints

#### Stream Reference Video

```http
GET /videos/{filename}
```

**Behavior:**
- If `VIDEOS_DIR` is set: Serves file from local directory
- If `VIDEOS_BASE_URL` is set: Redirects to CDN URL
- Otherwise: Returns 404

### 3.4 Inference Endpoints

#### Primary Inference (Retrieval)

```http
POST /v1/inference
Content-Type: application/json
```

**Request:**
```json
{
  "frames": [[0.01, 0.02, ...], ...],  // (T, 540) array
  "top_k": 5,
  "metadata": {
    "fps": 30,
    "source": "webcam"
  }
}
```

**Response:**
```json
{
  "top_k": 5,
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

#### Legacy Prediction (Single Result)

```http
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "pose_data": [[0.01, 0.02, ...], ...]  // (T, 540) array
}
```

**Response:**
```json
{
  "prediction": "THANK YOU"
}
```

---

## 4. Inference Service

### 4.1 Service Initialization

```python
class OnnxSignInferenceService:
    def __init__(self, settings: Settings) -> None:
        # Configure providers
        if settings.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        # Create ONNX session with optimizations
        session_opts = self._build_session_options()
        self._session = ort.InferenceSession(
            str(settings.onnx_model_path),
            providers=providers,
            sess_options=session_opts,
        )
        
        # Load artifacts
        self._global_mean, self._global_std = self._load_global_stats()
        self._embeddings = self._load_reference_embeddings()
        self._mapping = pd.read_csv(settings.mapping_path)
        self._proto_lookup = self._load_prototypes()
        
        # Warmup
        self._warmup()
```

### 4.2 Prediction Flow

```python
def predict(self, frames: Iterable[Iterable[float]], top_k: int = None) -> List[Prediction]:
    # 1. Convert to numpy
    frame_array = self._to_numpy(frames)
    
    # 2. Compute and validate motion
    motion = compute_motion_energy(frame_array)
    motion_mean = float(motion.mean())
    if motion_mean < self._settings.motion_floor:
        raise ValueError(f"Insufficient motion ({motion_mean:.4e})")
    
    # 3. Prepare sequence (normalize, pad/crop)
    norm_seq = self._prepare_sequence(frame_array)
    
    # 4. Run model
    embedding = self._run_model(norm_seq[np.newaxis, ...])
    
    # 5. Score neighbors
    refined, raw, probs, dtw_meta = self._score_neighbours(norm_seq, embedding)
    
    # 6. Format predictions
    return self._format_predictions(refined, raw, probs, dtw_meta, embedding, motion_mean, top_k)
```

### 4.3 Session Options

```python
@staticmethod
def _build_session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.enable_mem_pattern = False
    opts.enable_cpu_mem_arena = True
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts
```

### 4.4 Thread Safety

```python
def _run_model(self, sequence: np.ndarray) -> np.ndarray:
    with self._lock:  # Thread-safe access
        outputs = self._session.run([self._output_name], {self._input_name: sequence})
    
    embedding = outputs[0]
    if embedding.ndim == 2:
        embedding = embedding[0]
    
    # L2 normalize
    norm = np.linalg.norm(embedding) + 1e-8
    return embedding / norm
```

---

## 5. Request/Response Schemas

### 5.1 Request Models

```python
class SequencePayload(BaseModel):
    frames: List[List[float]]
    top_k: Optional[int] = None
    metadata: Optional[dict] = None

class PredictionRequest(BaseModel):
    pose_data: List[List[float]]
```

### 5.2 Response Models

```python
class Prediction(BaseModel):
    sentence_id: int
    sentence: Optional[str]
    category: Optional[str]
    video_file: Optional[str]
    similarity: float
    confidence: float
    raw_similarity: float
    proto_similarity: Optional[float]
    motion_mean: float
    accepted: bool
    rejection_reason: Optional[str]
    dtw_distance: Optional[float]
    dtw_score: Optional[float]

class InferenceResponse(BaseModel):
    top_k: int
    results: List[Prediction]

class HealthResponse(BaseModel):
    status: str
    detail: str

class ConfigResponse(BaseModel):
    seq_len: int
    feature_dim: int
    default_top_k: int
    model_provider: str
    motion_floor: float
    sim_threshold: float
    sim_margin: float
    confidence_temperature: float
    dtw_enabled: bool
```

---

## 6. Dependency Injection

### 6.1 Service Dependencies

```python
# dependencies.py
_service_instance: Optional[OnnxSignInferenceService] = None

def get_inference_service() -> OnnxSignInferenceService:
    global _service_instance
    if _service_instance is None:
        settings = get_settings()
        _service_instance = OnnxSignInferenceService(settings)
    return _service_instance

def get_service_settings() -> Settings:
    return get_settings()
```

### 6.2 Using Dependencies in Routes

```python
@app.get("/healthz", response_model=HealthResponse)
async def health_check(
    service: OnnxSignInferenceService = Depends(get_inference_service)
) -> HealthResponse:
    return HealthResponse(status="ok", detail=f"provider={service.provider}")

@app.post("/v1/inference", response_model=InferenceResponse)
async def run_inference(
    payload: SequencePayload,
    service: OnnxSignInferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    predictions = service.predict(payload.frames, top_k=payload.top_k)
    return InferenceResponse(top_k=len(predictions), results=predictions)
```

---

## 7. Middleware Configuration

### 7.1 CORS

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 7.2 Startup Event

```python
@app.on_event("startup")
async def startup_event() -> None:
    # Warm up the model so first request doesn't pay initialization cost
    get_inference_service()
```

---

## 8. Deployment

### 8.1 Local Development

```bash
# From project root
cd backend
pip install -r requirements.txt

# Run with uvicorn
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 8.2 Production (Render)

**render.yaml:**
```yaml
services:
  - type: web
    name: ghsl-backend
    runtime: python
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: MODELS_DIR
        value: /opt/render/project/src/backend/models
      - key: GLOBAL_STATS_PATH
        value: /opt/render/project/src/backend/models/global_stats.npz
      - key: MOTION_FLOOR
        value: 0.00075
      - key: ENABLE_CUDA
        value: false
```

### 8.3 Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY runs/ ./runs/

ENV MODELS_DIR=/app/runs
ENV PORT=8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 9. Error Handling

### 9.1 HTTP Exceptions

```python
# Validation error (400)
if pose_arr.shape[1] != settings.feature_dim:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid shape: {pose_arr.shape}, expected (T, {settings.feature_dim})"
    )

# Not found (404)
if not file_path.exists():
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Video not found"
    )

# Service unavailable (503)
if model_load_error:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc)
    )
```

### 9.2 Motion Rejection

```python
try:
    predictions = service.predict(payload.frames, top_k=payload.top_k)
except ValueError as exc:
    # Motion too low, etc.
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )
```

---

## 10. Monitoring & Debugging

### 10.1 Health Check Response

```json
{
  "status": "ok",
  "detail": "provider=CPUExecutionProvider"
}
```

### 10.2 Config Endpoint

Use `/v1/config` to verify runtime configuration:
- Model provider (CPU/CUDA)
- Threshold values
- DTW status

### 10.3 Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(self, frames, top_k):
    logger.info(f"Inference request: {len(frames)} frames, top_k={top_k}")
    # ...
    logger.info(f"Prediction complete: accepted={accepted}, similarity={top1_score:.4f}")
```

---

## 11. Requirements

```
# backend/requirements.txt
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
numpy>=1.24.0
pandas>=2.0.0
onnxruntime>=1.16.0
pydantic>=2.0.0
python-multipart>=0.0.6
aiofiles>=23.0.0

# Optional
fastdtw>=0.3.4  # For DTW re-ranking
onnxruntime-gpu>=1.16.0  # For CUDA support
```

---

## 12. Future Enhancements

- [ ] WebSocket support for streaming inference
- [ ] Batch inference endpoint for offline processing
- [ ] Rate limiting for public deployments
- [ ] OpenTelemetry tracing integration
- [ ] Model versioning and A/B testing
- [ ] Redis caching for embedding lookups
