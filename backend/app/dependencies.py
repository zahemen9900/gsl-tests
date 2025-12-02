from __future__ import annotations

from functools import lru_cache

from .config import Settings, get_settings
from .services.inference import OnnxSignInferenceService


@lru_cache(maxsize=1)
def get_inference_service() -> OnnxSignInferenceService:
    settings = get_settings()
    return OnnxSignInferenceService(settings)


def get_service_settings() -> Settings:
    return get_settings()
