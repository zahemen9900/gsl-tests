from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from .dependencies import get_inference_service, get_service_settings
from .routers import inference as realtime_router
from .schemas import (
    ConfigResponse,
    HealthResponse,
    InferenceResponse,
    PredictionRequest,
    PredictionResponse,
    SampleItem,
    SampleMetadata,
    SequencePayload,
)
from .services.inference import OnnxSignInferenceService

app = FastAPI(
    title="GhSL Local Backend",
    version="0.2.0",
    description="Unified backend with ONNX retrieval + legacy compatibility endpoints.",
)

app.include_router(realtime_router.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    # warm up provider so first request does not pay initialization cost
    get_inference_service()


@app.get("/healthz", response_model=HealthResponse, tags=["system"])
async def health_check(service: OnnxSignInferenceService = Depends(get_inference_service)) -> HealthResponse:
    try:
        provider = service.provider
        return HealthResponse(status="ok", detail=f"provider={provider}")
    except Exception as exc:  # pragma: no cover - defensive path
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@app.get("/v1/config", response_model=ConfigResponse, tags=["metadata"])
async def get_config() -> ConfigResponse:
    settings = get_service_settings()
    service = get_inference_service()
    return ConfigResponse(
        seq_len=settings.seq_len,
        feature_dim=settings.feature_dim,
        default_top_k=settings.top_k,
        model_provider=service.provider,
        motion_floor=settings.motion_floor,
        sim_threshold=settings.sim_threshold,
        sim_margin=settings.sim_margin,
        confidence_temperature=settings.confidence_temperature,
        dtw_enabled=service.dtw_active,
    )


@app.get("/v1/samples/random", response_model=SampleMetadata, tags=["metadata"])
async def random_sample(service: OnnxSignInferenceService = Depends(get_inference_service)) -> SampleMetadata:
    settings = get_service_settings()
    record = service.random_sample()
    video_file = record.get("video_file")
    video_url = None
    if video_file:
        if settings.videos_dir:
            video_url = f"/videos/{video_file}"
        elif settings.videos_base_url:
            video_url = f"{settings.videos_base_url}/{video_file}"
    return SampleMetadata(
        sentence_id=int(record.get("sentence_id", -1)),
        sentence=record.get("sentence"),
        category=record.get("category"),
        video_file=video_file,
        video_url=video_url,
    )


@app.get("/videos/{filename}", tags=["media"])
async def stream_reference_video(filename: str) -> Response:
    settings = get_service_settings()
    safe_name = Path(filename).name
    if settings.videos_dir:
        file_path = settings.videos_dir / safe_name
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
        if file_path.suffix.lower() != ".mp4":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported media type")
        return FileResponse(file_path, media_type="video/mp4", filename=safe_name)
    if settings.videos_base_url:
        remote_url = f"{settings.videos_base_url}/{safe_name}"
        return RedirectResponse(url=remote_url)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video streaming disabled")


@app.post("/v1/inference", response_model=InferenceResponse, tags=["inference"])
async def run_inference(
    payload: SequencePayload,
    service: OnnxSignInferenceService = Depends(get_inference_service),
) -> InferenceResponse:
    try:
        predictions = service.predict(payload.frames, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return InferenceResponse(top_k=len(predictions), results=predictions)


# ---------------------------------------------------------------------------
# Legacy-compatible endpoints for the current frontend
# ---------------------------------------------------------------------------


@app.get("/samples", response_model=List[SampleItem])
async def get_samples(service: OnnxSignInferenceService = Depends(get_inference_service)) -> List[SampleItem]:
    samples = service.sample_batch(count=5)
    return [
        SampleItem(
            video_file=record.get("video_file", ""),
            sentence=record.get("sentence", ""),
            feature_path=record.get("feature_path", ""),
        )
        for record in samples
    ]


@app.post("/predict", response_model=PredictionResponse)
async def legacy_predict(
    request: PredictionRequest,
    service: OnnxSignInferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    pose_arr = np.asarray(request.pose_data, dtype=np.float32)
    if pose_arr.ndim != 2 or pose_arr.shape[1] != get_service_settings().feature_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid shape: {pose_arr.shape}, expected (T, {get_service_settings().feature_dim})",
        )
    try:
        predictions = service.predict(pose_arr, top_k=1)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not predictions:
        return PredictionResponse(prediction="")
    return PredictionResponse(prediction=predictions[0].sentence or "")
