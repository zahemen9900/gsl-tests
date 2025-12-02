from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pose_data: List[List[float]]


class PredictionResponse(BaseModel):
    prediction: str


class SampleItem(BaseModel):
    video_file: str
    sentence: str
    feature_path: str


class SequencePayload(BaseModel):
    frames: List[List[float]] = Field(
        ..., min_length=1, description="Sequence of per-frame landmark feature vectors"
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Requested number of retrieval results")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata such as fps or capture source")


class Prediction(BaseModel):
    sentence_id: int = Field(..., description="Identifier for the matched sentence")
    sentence: Optional[str] = Field(default=None, description="Human-readable sentence for the match")
    category: Optional[str] = Field(default=None, description="Sign category label")
    video_file: Optional[str] = Field(default=None, description="Reference video file for the match")
    similarity: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Heuristic confidence score derived from similarity")
    raw_similarity: Optional[float] = Field(default=None, description="Similarity prior to any re-ranking adjustments")
    proto_similarity: Optional[float] = Field(
        default=None, description="Similarity between the query embedding and the sentence prototype"
    )
    motion_mean: Optional[float] = Field(default=None, description="Mean motion energy observed in the query clip")
    accepted: Optional[bool] = Field(default=None, description="Whether gating thresholds accepted the top prediction")
    rejection_reason: Optional[str] = Field(default=None, description="Rejection reason when gating fails")
    dtw_distance: Optional[float] = Field(default=None, description="DTW distance used during re-ranking, if available")
    dtw_score: Optional[float] = Field(default=None, description="DTW-derived score blended into similarity, if available")


class InferenceResponse(BaseModel):
    top_k: int
    results: List[Prediction]


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status string")
    detail: Optional[str] = Field(default=None, description="Additional diagnostic information")


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


class SampleMetadata(BaseModel):
    sentence_id: int
    sentence: Optional[str] = None
    category: Optional[str] = None
    video_file: Optional[str] = None
    video_url: Optional[str] = None
