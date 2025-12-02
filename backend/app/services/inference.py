import math
import random
import threading
from numbers import Integral
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import onnxruntime as ort
import pandas as pd

try:  # Optional DTW re-ranking support
    from fastdtw import fastdtw
except ImportError:  # pragma: no cover - DTW optional
    fastdtw = None

from ..config import Settings
from ..schemas import Prediction


def normalize_with_stats(seq: np.ndarray, mean_vec: np.ndarray, std_vec: np.ndarray) -> np.ndarray:
    return (seq - mean_vec) / (std_vec + 1e-6)


def center_time_crop(seq: np.ndarray, target_len: int) -> np.ndarray:
    total = seq.shape[0]
    if total <= target_len:
        return seq
    start = (total - target_len) // 2
    return seq[start:start + target_len]


def compute_motion_energy(frames: np.ndarray) -> np.ndarray:
    if frames.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    pose_xyz = frames[:, :33 * 4].reshape(frames.shape[0], 33, 4)[..., :3]
    pose_xyz = pose_xyz.reshape(frames.shape[0], -1)
    hands = frames[:, 33 * 4:33 * 4 + 2 * 21 * 3]
    face = frames[:, 33 * 4 + 2 * 21 * 3:]
    coords = np.concatenate([pose_xyz, hands, face], axis=1)
    diffs = np.diff(coords, axis=0, prepend=coords[:1])
    return np.linalg.norm(diffs, axis=1).astype(np.float32)


class OnnxSignInferenceService:
    """Wrap ONNX Runtime inference, gating, and nearest-neighbour retrieval."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._min_input_frames = max(8, min(settings.seq_len, 16))

        providers: List[str]
        if settings.use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        session_opts = self._build_session_options()
        try:
            self._session = ort.InferenceSession(
                str(settings.onnx_model_path),
                providers=providers,
                sess_options=session_opts,
            )
        except Exception:
            if settings.use_gpu:
                self._session = ort.InferenceSession(
                    str(settings.onnx_model_path),
                    providers=["CPUExecutionProvider"],
                    sess_options=session_opts,
                )
            else:
                raise

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._provider = self._session.get_providers()[0]

        output_info = self._session.get_outputs()[0]
        last_dim = output_info.shape[-1] if output_info.shape else None
        embedding_dim = int(last_dim) if isinstance(last_dim, Integral) else None
        self._embedding_dim: Optional[int] = embedding_dim

        self._global_mean, self._global_std = self._load_global_stats(settings.global_stats_path)
        self._embeddings = self._load_reference_embeddings(settings.embeddings_path)
        self._mapping = pd.read_csv(settings.mapping_path)
        if len(self._embeddings) != len(self._mapping):
            raise ValueError("Embeddings and mapping file have mismatched lengths")
        self._mapping_records = self._mapping.to_dict(orient="records")
        if not self._mapping_records:
            raise ValueError("Mapping file does not contain any entries")

        self._proto_lookup = self._load_prototypes(settings.prototypes_path)
        self._dtw_enabled = settings.dtw_enabled and fastdtw is not None

        self._warmup()

    @staticmethod
    def _build_session_options() -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        opts.enable_cpu_mem_arena = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    @property
    def config(self) -> Settings:
        return self._settings

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def dtw_active(self) -> bool:
        return self._dtw_enabled

    def random_sample(self) -> dict:
        return random.choice(self._mapping_records).copy()

    def sample_batch(self, count: int = 5) -> List[dict]:
        if count <= 0:
            return []
        count = min(count, len(self._mapping_records))
        return random.sample(self._mapping_records, count)

    def _warmup(self) -> None:
        dummy_frames = self._settings.seq_len
        dummy = np.zeros((1, dummy_frames, self._settings.feature_dim), dtype=np.float32)
        with self._lock:
            self._session.run([self._output_name], {self._input_name: dummy})

    def predict(self, frames: Iterable[Iterable[float]], top_k: int | None = None) -> List[Prediction]:
        frame_array = self._to_numpy(frames)
        motion = compute_motion_energy(frame_array)
        motion_mean = float(motion.mean()) if motion.size else 0.0
        if motion_mean < self._settings.motion_floor:
            raise ValueError(
                f"Insufficient motion ({motion_mean:.4e} < {self._settings.motion_floor:.4e}); clip rejected."
            )

        norm_seq = self._prepare_sequence(frame_array, expand=False)
        embedding = self._run_model(norm_seq[np.newaxis, ...])
        refined, raw, probs, dtw_meta = self._score_neighbours(norm_seq, embedding)
        return self._format_predictions(
            refined,
            raw,
            probs,
            dtw_meta,
            embedding,
            motion_mean,
            top_k,
        )

    def _to_numpy(self, frames: Iterable[Iterable[float]]) -> np.ndarray:
        data = np.asarray(list(frames), dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("Frames input must produce a 2-D matrix")
        if data.shape[1] != self._settings.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self._settings.feature_dim}, received {data.shape[1]}"
            )
        return data

    def _prepare_sequence(self, seq: np.ndarray, *, expand: bool = True) -> np.ndarray:
        data = seq.astype(np.float32, copy=True)
        if data.shape[0] == 0:
            raise ValueError("Sequence contains no frames")
        if data.shape[0] > self._settings.seq_len:
            data = center_time_crop(data, self._settings.seq_len)
        data = normalize_with_stats(data, self._global_mean, self._global_std)
        if data.shape[0] < self._min_input_frames:
            pad = np.repeat(data[-1:], self._min_input_frames - data.shape[0], axis=0)
            data = np.vstack([data, pad])
        if data.shape[0] < self._settings.seq_len:
            pad = np.repeat(data[-1:], self._settings.seq_len - data.shape[0], axis=0)
            data = np.vstack([data, pad])
        if expand:
            return np.expand_dims(data, axis=0)
        return data

    def _run_model(self, sequence: np.ndarray) -> np.ndarray:
        with self._lock:
            outputs = self._session.run([self._output_name], {self._input_name: sequence})
        embedding = outputs[0]
        if embedding.ndim == 2:
            embedding = embedding[0]
        elif embedding.ndim != 1:
            raise ValueError("Unexpected embedding shape returned from model")
        if self._embedding_dim is None:
            self._embedding_dim = int(embedding.shape[0])
        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def _score_neighbours(
        self,
        query_seq: np.ndarray,
        embedding: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, float]]]:
        raw_scores = self._embeddings @ embedding
        refined_scores = raw_scores.copy()
        dtw_meta: dict[int, dict[str, float]] = {}

        if self._dtw_enabled:
            assert fastdtw is not None
            top_candidates = refined_scores.argsort()[::-1][: self._settings.dtw_topk]
            for idx in top_candidates:
                ref_seq = self._load_reference_sequence(idx)
                if ref_seq is None:
                    continue
                dist, _ = fastdtw(
                    query_seq,
                    ref_seq,
                    radius=self._settings.dtw_radius,
                    dist=lambda x, y: float(np.linalg.norm(x - y)),
                )
                dtw_score = math.exp(-self._settings.dtw_lambda * dist)
                refined_scores[idx] = (
                    self._settings.dtw_alpha * refined_scores[idx]
                    + (1.0 - self._settings.dtw_alpha) * dtw_score
                )
                dtw_meta[idx] = {"dtw_distance": float(dist), "dtw_score": float(dtw_score)}

        logits = refined_scores / self._settings.confidence_temperature
        probs = self._softmax(logits)
        return refined_scores, raw_scores, probs, dtw_meta

    def _load_reference_sequence(self, index: int) -> Optional[np.ndarray]:
        row = self._mapping.iloc[int(index)]
        feature_path = row.get("feature_path")
        if not feature_path or not Path(feature_path).exists():
            return None
        arr = np.load(feature_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._settings.feature_dim:
            return None
        return self._prepare_sequence(arr, expand=False)

    def _format_predictions(
        self,
        refined: np.ndarray,
        raw: np.ndarray,
        probs: np.ndarray,
        dtw_meta: dict[int, dict[str, float]],
        embedding: np.ndarray,
        motion_mean: float,
        top_k: Optional[int],
    ) -> List[Prediction]:
        requested_top_k = top_k or self._settings.top_k
        requested_top_k = max(1, min(int(requested_top_k), len(self._embeddings)))

        indices = refined.argsort()[::-1][:requested_top_k]
        if indices.size == 0:
            return []

        top1_idx = int(indices[0])
        top1 = float(refined[top1_idx])
        top2 = float(refined[int(indices[1])]) if indices.size > 1 else -1.0
        top_sentence_id = int(self._mapping.iloc[top1_idx].get("sentence_id", -1))
        proto_sim_top = None
        if top_sentence_id in self._proto_lookup:
            proto_sim_top = float(np.dot(self._proto_lookup[top_sentence_id], embedding))

        similarity_ok = top1 >= self._settings.sim_threshold
        margin_ok = (top1 - top2) >= self._settings.sim_margin if top2 > -1.0 else True
        proto_ok = True
        if proto_sim_top is not None:
            proto_ok = proto_sim_top >= (self._settings.sim_threshold - 0.05)

        if similarity_ok and margin_ok and proto_ok:
            accepted = True
            reason = "ok"
        elif not similarity_ok:
            accepted = False
            reason = "low_similarity"
        elif not margin_ok:
            accepted = False
            reason = "ambiguous_top2"
        else:
            accepted = False
            reason = "prototype_disagreement"

        results: List[Prediction] = []
        for idx in indices:
            row = self._mapping.iloc[int(idx)]
            sentence_id = int(row.get("sentence_id", -1))
            proto_sim = None
            if sentence_id in self._proto_lookup:
                proto_sim = float(np.dot(self._proto_lookup[sentence_id], embedding))
            meta = dtw_meta.get(int(idx), {})
            similarity = float(refined[int(idx)])
            raw_similarity = float(raw[int(idx)])
            confidence = float(np.clip(probs[int(idx)], 0.0, 1.0))

            results.append(
                Prediction(
                    sentence_id=sentence_id,
                    sentence=row.get("sentence"),
                    category=row.get("category"),
                    video_file=row.get("video_file"),
                    similarity=similarity,
                    confidence=confidence,
                    raw_similarity=raw_similarity,
                    proto_similarity=proto_sim,
                    motion_mean=motion_mean,
                    accepted=accepted,
                    rejection_reason=None if accepted else reason,
                    dtw_distance=meta.get("dtw_distance"),
                    dtw_score=meta.get("dtw_score"),
                )
            )
        return results

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        offset = logits - logits.max()
        exp_vals = np.exp(offset)
        denom = exp_vals.sum()
        if denom <= 0:
            return np.zeros_like(logits)
        return exp_vals / denom

    def _load_reference_embeddings(self, path: str | Path) -> np.ndarray:
        data = np.load(path)
        if data.ndim != 2:
            raise ValueError("Embeddings array must be 2-D")
        if self._embedding_dim is None:
            self._embedding_dim = int(data.shape[1])
        elif data.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Embeddings dimension mismatch: expected {self._embedding_dim}, received {data.shape[1]}"
            )
        norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-8
        return data / norms

    def _load_global_stats(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        blob = np.load(path)
        mean = blob["feature_mean"].astype(np.float32)
        std = blob["feature_std"].astype(np.float32)
        std = np.where(std == 0, 1.0, std)
        return mean, std

    def _load_prototypes(self, path: Optional[str | Path]) -> dict[int, np.ndarray]:
        if not path:
            return {}
        proto_path = Path(path)
        if not proto_path.exists():
            return {}
        blob = np.load(proto_path)
        lookup: dict[int, np.ndarray] = {}
        for sid, emb in zip(blob["sentence_ids"], blob["embeddings"]):
            vec = emb / (np.linalg.norm(emb) + 1e-8)
            if self._embedding_dim is not None and vec.shape[0] != self._embedding_dim:
                raise ValueError(
                    f"Prototype embedding dimension mismatch: expected {self._embedding_dim}, received {vec.shape[0]}"
                )
            lookup[int(sid)] = vec.astype(np.float32)
        return lookup
