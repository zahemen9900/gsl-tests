from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the FastAPI service."""

    seq_len: int
    feature_dim: int
    top_k: int
    models_dir: Path
    onnx_model_path: Path
    embeddings_path: Path
    mapping_path: Path
    prototypes_path: Optional[Path]
    global_stats_path: Path
    use_gpu: bool
    videos_dir: Optional[Path]
    videos_base_url: Optional[str]
    motion_floor: float
    sim_threshold: float
    sim_margin: float
    confidence_temperature: float
    dtw_enabled: bool
    dtw_radius: int
    dtw_alpha: float
    dtw_lambda: float
    dtw_topk: int

    @staticmethod
    def _bool_from_env(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(os.getenv("BACKEND_BASE_DIR", Path(__file__).resolve().parent.parent)).resolve()
        models_dir = Path(os.getenv("MODELS_DIR", base_dir.parent / "models")).resolve()
        onnx_model = Path(os.getenv("ONNX_MODEL_PATH", models_dir / "sign_encoder.onnx")).resolve()
        embeddings_path = Path(os.getenv("EMBEDDINGS_PATH", models_dir / "embeddings.npy")).resolve()
        mapping_path = Path(os.getenv("MAPPING_PATH", models_dir / "embeddings_map.csv")).resolve()
        prototypes_env = os.getenv("PROTOTYPES_PATH", models_dir / "prototypes.npz")
        prototypes_path = Path(prototypes_env).expanduser().resolve() if prototypes_env else None

        default_videos_dir = base_dir.parent / "sample_dataset" / "videos"
        videos_dir_env = os.getenv("VIDEOS_DIR")
        if videos_dir_env:
            videos_dir_candidate: Optional[Path] = Path(videos_dir_env).expanduser().resolve()
        elif default_videos_dir.exists():
            videos_dir_candidate = default_videos_dir.resolve()
        else:
            videos_dir_candidate = None

        global_stats_env = os.getenv("GLOBAL_STATS_PATH")
        if global_stats_env:
            global_stats_path = Path(global_stats_env).expanduser().resolve()
        else:
            stats_candidates = [
                models_dir / "global_stats.npz",
                base_dir.parent / "proc" / "global_stats.npz",
            ]
            global_stats_path = None
            for candidate in stats_candidates:
                resolved = candidate.resolve()
                if resolved.exists():
                    global_stats_path = resolved
                    break
            if global_stats_path is None:
                global_stats_path = stats_candidates[0].resolve()

        videos_base_url_env = os.getenv("VIDEOS_BASE_URL")
        videos_base_url = videos_base_url_env.rstrip("/") if videos_base_url_env else None

        seq_len = int(os.getenv("SEQ_LEN", 64))
        feature_dim = int(os.getenv("FEATURE_DIM", 540))
        top_k = int(os.getenv("DEFAULT_TOP_K", 5))
        use_gpu = cls._bool_from_env("ENABLE_CUDA", False)
        motion_floor = float(os.getenv("MOTION_FLOOR", 7.5e-4))
        sim_threshold = float(os.getenv("SIM_THRESHOLD", 0.58))
        sim_margin = float(os.getenv("SIM_MARGIN", 0.10))
        confidence_temperature = float(os.getenv("CONFIDENCE_TEMPERATURE", 0.12))
        dtw_enabled = cls._bool_from_env("DTW_ENABLED", True)
        dtw_radius = int(os.getenv("DTW_RADIUS", 6))
        dtw_alpha = float(os.getenv("DTW_ALPHA", 0.65))
        dtw_lambda = float(os.getenv("DTW_LAMBDA", 0.002))
        dtw_topk = int(os.getenv("DTW_TOPK", 5))

        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        for path in (onnx_model, embeddings_path, mapping_path):
            if not path.exists():
                raise FileNotFoundError(f"Required model artifact missing: {path}")
        if not global_stats_path.exists():
            raise FileNotFoundError(f"Global stats file not found: {global_stats_path}")
        if prototypes_path and not prototypes_path.exists():
            prototypes_path = None

        if videos_dir_candidate and not videos_dir_candidate.exists():
            raise FileNotFoundError(f"Reference video directory not found: {videos_dir_candidate}")
        videos_dir = videos_dir_candidate

        if not videos_dir and not videos_base_url:
            print("Warning: no local video directory or base URL configured; /videos endpoint will be disabled.")

        return cls(
            seq_len=seq_len,
            feature_dim=feature_dim,
            top_k=top_k,
            models_dir=models_dir,
            onnx_model_path=onnx_model,
            embeddings_path=embeddings_path,
            mapping_path=mapping_path,
            prototypes_path=prototypes_path,
            global_stats_path=global_stats_path,
            use_gpu=use_gpu,
            videos_dir=videos_dir,
            videos_base_url=videos_base_url,
            motion_floor=motion_floor,
            sim_threshold=sim_threshold,
            sim_margin=sim_margin,
            confidence_temperature=confidence_temperature,
            dtw_enabled=dtw_enabled,
            dtw_radius=dtw_radius,
            dtw_alpha=dtw_alpha,
            dtw_lambda=dtw_lambda,
            dtw_topk=dtw_topk,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Provide cached Settings so env parsing happens once."""

    return Settings.from_env()
