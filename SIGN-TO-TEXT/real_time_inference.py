#!/usr/bin/env python3
"""Real-time GhSL sentence retrieval demo using the refreshed encoder stack."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:  # Optional DTW re-ranking support
    from fastdtw import fastdtw
except ImportError:  # pragma: no cover - DTW optional
    fastdtw = None


# -----------------------------------------------------------------------------
# Geometry/feature helpers shared with the preprocessing notebook
# -----------------------------------------------------------------------------
FACE_DOWNSAMPLE = 5
NUM_FACE_LANDMARKS = len(range(0, 468, FACE_DOWNSAMPLE))  # = 94
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
PER_FRAME_FEATURES = NUM_POSE_LANDMARKS * 4 + 2 * NUM_HAND_LANDMARKS * 3 + NUM_FACE_LANDMARKS * 3

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def normalize_frame_landmarks(frame_feats: np.ndarray) -> np.ndarray:
    """Center and scale landmarks around the torso to reduce camera variance."""
    feats = np.asarray(frame_feats, dtype=np.float32)

    pose = feats[: NUM_POSE_LANDMARKS * 4].reshape(NUM_POSE_LANDMARKS, 4)
    left_hand_start = NUM_POSE_LANDMARKS * 4
    left_hand_end = left_hand_start + NUM_HAND_LANDMARKS * 3
    right_hand_end = left_hand_end + NUM_HAND_LANDMARKS * 3

    left_hand = feats[left_hand_start:left_hand_end].reshape(NUM_HAND_LANDMARKS, 3)
    right_hand = feats[left_hand_end:right_hand_end].reshape(NUM_HAND_LANDMARKS, 3)
    face = feats[right_hand_end:right_hand_end + NUM_FACE_LANDMARKS * 3].reshape(NUM_FACE_LANDMARKS, 3)

    pose_coords = pose[:, :3].copy()
    torso_pts = pose_coords[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]]

    if not np.isfinite(torso_pts).all() or np.linalg.norm(torso_pts) < 1e-6:
        return feats.astype(np.float32)

    torso_center = torso_pts.mean(axis=0)
    pose_coords -= torso_center
    left_hand -= torso_center
    right_hand -= torso_center
    face -= torso_center

    shoulder_span = np.linalg.norm(pose_coords[LEFT_SHOULDER] - pose_coords[RIGHT_SHOULDER])
    hip_span = np.linalg.norm(pose_coords[LEFT_HIP] - pose_coords[RIGHT_HIP])
    torso_height = np.linalg.norm(
        0.5 * (pose_coords[LEFT_SHOULDER] + pose_coords[RIGHT_SHOULDER]) -
        0.5 * (pose_coords[LEFT_HIP] + pose_coords[RIGHT_HIP])
    )

    candidates = [c for c in (shoulder_span, hip_span, torso_height) if c > 1e-4]
    scale = float(np.median(candidates)) if candidates else 1.0

    pose_coords /= scale
    left_hand /= scale
    right_hand /= scale
    face /= scale

    pose[:, :3] = pose_coords

    normalized = np.concatenate([
        pose.reshape(-1),
        left_hand.reshape(-1),
        right_hand.reshape(-1),
        face.reshape(-1)
    ]).astype(np.float32)

    return normalized


def extract_feature_vector(results: mp.solutions.holistic.HolisticResults) -> Optional[np.ndarray]:
    """Convert MediaPipe holistic results to the flattened feature vector."""
    frame_feats: List[float] = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            frame_feats.extend([lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)])
    else:
        frame_feats.extend([0.0] * (NUM_POSE_LANDMARKS * 4))

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_feats.extend([lm.x, lm.y, lm.z])
    else:
        frame_feats.extend([0.0] * (NUM_HAND_LANDMARKS * 3))

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_feats.extend([lm.x, lm.y, lm.z])
    else:
        frame_feats.extend([0.0] * (NUM_HAND_LANDMARKS * 3))

    if results.face_landmarks:
        for idx in range(0, 468, FACE_DOWNSAMPLE):
            lm = results.face_landmarks.landmark[idx]
            frame_feats.extend([lm.x, lm.y, lm.z])
    else:
        frame_feats.extend([0.0] * (NUM_FACE_LANDMARKS * 3))

    if len(frame_feats) != PER_FRAME_FEATURES:
        return None

    return normalize_frame_landmarks(np.asarray(frame_feats, dtype=np.float32))


# -----------------------------------------------------------------------------
# Runtime model config helpers
# -----------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    seq_len: int = 64
    input_dim: int = PER_FRAME_FEATURES
    proj_dim: int = 160
    rnn_hidden: int = 160
    rnn_layers: int = 2
    attn_heads: int = 4
    embed_dim: int = 256
    motion_floor: float = 7.5e-4
    sim_threshold: float = 0.58
    sim_margin: float = 0.10
    confidence_temperature: float = 0.12
    prototype_min_clips: int = 2
    dtw_enabled: bool = True
    dtw_radius: int = 6
    dtw_alpha: float = 0.65
    dtw_lambda: float = 0.002
    dtw_topk: int = 5

    @classmethod
    def from_training_cfg(cls, cfg: Dict[str, object]) -> "RuntimeConfig":
        data = {}
        for field in cls.__dataclass_fields__:
            if field in cfg and not isinstance(cfg[field], torch.device):
                data[field] = cfg[field]
        dtw_cfg = cfg.get("dtw", {}) if isinstance(cfg.get("dtw"), dict) else {}
        if isinstance(dtw_cfg, dict):
            data.setdefault("dtw_enabled", bool(dtw_cfg.get("enabled", cls.dtw_enabled)))
            data.setdefault("dtw_radius", dtw_cfg.get("radius", cls.dtw_radius))
            data.setdefault("dtw_alpha", dtw_cfg.get("alpha", cls.dtw_alpha))
            data.setdefault("dtw_lambda", dtw_cfg.get("lambda", cls.dtw_lambda))
            data.setdefault("dtw_topk", dtw_cfg.get("topk", cls.dtw_topk))
        return cls(**data)


# -----------------------------------------------------------------------------
# Model definitions (mirrors training notebook)
# -----------------------------------------------------------------------------


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


class FrameProjector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 160) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.act(self.fc2(out))
        return out


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 160, n_layers: int = 2, embed_dim: int = 256, attn_heads: int = 4) -> None:
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.attn_heads = attn_heads
        self.attn_key = nn.Linear(hidden * 2, hidden)
        self.attn_score = nn.Linear(hidden, attn_heads)
        self.attn_proj = nn.Linear(hidden * 2, embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_out, _ = self.gru(x)
        keys = torch.tanh(self.attn_key(seq_out))
        attn_logits = self.attn_score(keys)

        motion = torch.norm(x[:, 1:] - x[:, :-1], dim=-1, keepdim=True)
        motion = torch.cat([motion[:, :1], motion], dim=1)
        motion = (motion - motion.mean(dim=1, keepdim=True)) / (motion.std(dim=1, keepdim=True) + 1e-6)
        attn_logits = attn_logits + motion

        weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(seq_out.unsqueeze(-2) * weights.unsqueeze(-1), dim=1)
        pooled = pooled.mean(dim=1)

        embedding = self.attn_proj(self.dropout(pooled))
        embedding = self.post_norm(embedding)
        embedding = nn.functional.normalize(embedding, dim=-1)
        return embedding, weights.mean(dim=2)


class SignEmbeddingModel(nn.Module):
    def __init__(self, cfg: RuntimeConfig) -> None:
        super().__init__()
        self.projector = FrameProjector(cfg.input_dim, cfg.proj_dim)
        self.encoder = TemporalEncoder(cfg.proj_dim, cfg.rnn_hidden, cfg.rnn_layers, cfg.embed_dim, cfg.attn_heads)

    def forward(self, x: torch.Tensor, return_sequence: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj = self.projector(x)
        embedding, attn_weights = self.encoder(proj)
        if return_sequence:
            return embedding, proj, attn_weights
        return embedding


# -----------------------------------------------------------------------------
# Inference engine
# -----------------------------------------------------------------------------


class InferenceEngine:
    def __init__(
        self,
        model: SignEmbeddingModel,
        embeddings_path: str | Path,
        mapping_path: str | Path,
        cfg: RuntimeConfig,
        global_mean: np.ndarray,
        global_std: np.ndarray,
        device: torch.device,
        prototypes_path: Optional[str | Path] = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self.seq_len = cfg.seq_len
        self.global_mean = global_mean.astype(np.float32)
        self.global_std = np.where(global_std == 0, 1.0, global_std).astype(np.float32)

        self.embs = np.load(embeddings_path)
        mapping = pd.read_csv(mapping_path)
        if len(self.embs) != len(mapping):
            raise ValueError("Embeddings and mapping size mismatch")
        norms = np.linalg.norm(self.embs, axis=1, keepdims=True) + 1e-8
        self.embs = self.embs / norms
        self.mapping = mapping

        self.proto_lookup: Dict[int, np.ndarray] = {}
        if prototypes_path and Path(prototypes_path).exists():
            proto_blob = np.load(prototypes_path)
            for sid, emb in zip(proto_blob["sentence_ids"], proto_blob["embeddings"]):
                vec = emb / (np.linalg.norm(emb) + 1e-8)
                self.proto_lookup[int(sid)] = vec.astype(np.float32)

    def preprocess_sequence(self, feat_np: np.ndarray) -> np.ndarray:
        seq = feat_np.astype(np.float32)
        if seq.ndim != 2:
            raise ValueError("Expected a 2-D array of frames")
        if seq.shape[0] == 0:
            raise ValueError("Sequence is empty")
        if seq.shape[0] > self.seq_len:
            seq = center_time_crop(seq, self.seq_len)
        seq = normalize_with_stats(seq, self.global_mean, self.global_std)
        if seq.shape[0] < 8:
            pad = np.repeat(seq[-1:], 8 - seq.shape[0], axis=0)
            seq = np.vstack([seq, pad])
        return seq

    @torch.no_grad()
    def predict(self, feat_np: np.ndarray, topk: int = 5) -> Dict[str, object]:
        motion_vals = compute_motion_energy(feat_np.astype(np.float32))
        motion_mean = float(motion_vals.mean()) if motion_vals.size else 0.0
        if motion_mean < self.cfg.motion_floor:
            return {"accepted": False, "reason": "low_motion", "motion_mean": motion_mean, "top": []}

        seq_np = self.preprocess_sequence(feat_np)
        tensor = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)
        embedding, proj_seq, attn = self.model(tensor, return_sequence=True)
        emb_np = embedding.cpu().numpy()[0]
        emb_np = emb_np / (np.linalg.norm(emb_np) + 1e-8)
        proj_np = proj_seq.cpu().numpy()[0]
        attn_np = attn.cpu().numpy()[0]

        raw_sims = self.embs @ emb_np
        refined_sims = raw_sims.copy()
        dtw_meta: Dict[int, Dict[str, float]] = {}

        if self.cfg.dtw_enabled and fastdtw is not None:
            top_candidates = refined_sims.argsort()[::-1][: self.cfg.dtw_topk]
            for idx in top_candidates:
                ref_path = self.mapping.iloc[idx].get("feature_path")
                if not ref_path or not Path(ref_path).exists():
                    continue
                ref_seq = np.load(ref_path).astype(np.float32)
                if ref_seq.ndim != 2 or ref_seq.shape[1] != self.cfg.input_dim:
                    continue
                ref_seq = self.preprocess_sequence(ref_seq)
                ref_proj = self.model.projector(
                    torch.from_numpy(ref_seq).unsqueeze(0).to(self.device)
                ).cpu().numpy()[0]
                dist, _ = fastdtw(
                    proj_np,
                    ref_proj,
                    radius=self.cfg.dtw_radius,
                    dist=lambda x, y: np.linalg.norm(x - y),
                )
                dtw_score = math.exp(-self.cfg.dtw_lambda * dist)
                blended = self.cfg.dtw_alpha * refined_sims[idx] + (1.0 - self.cfg.dtw_alpha) * dtw_score
                refined_sims[idx] = blended
                dtw_meta[idx] = {"dtw_distance": float(dist), "dtw_score": float(dtw_score)}

        logits = refined_sims / self.cfg.confidence_temperature
        probs = np.exp(logits - logits.max())
        probs /= probs.sum() + 1e-8

        top_indices = refined_sims.argsort()[::-1][:topk]
        predictions: List[Dict[str, object]] = []
        top1 = refined_sims[top_indices[0]] if top_indices.size > 0 else -1.0
        top2 = refined_sims[top_indices[1]] if top_indices.size > 1 else -1.0

        for rank, idx in enumerate(top_indices, start=1):
            row = self.mapping.iloc[int(idx)]
            sentence_id = int(row.get("sentence_id", -1))
            proto_sim = None
            if sentence_id in self.proto_lookup:
                proto_sim = float(np.dot(self.proto_lookup[sentence_id], emb_np))
            entry = {
                "rank": rank,
                "video_file": row.get("video_file"),
                "feature_path": row.get("feature_path"),
                "sentence_id": sentence_id,
                "sentence": row.get("sentence"),
                "category": row.get("category"),
                "similarity": float(refined_sims[idx]),
                "raw_similarity": float(raw_sims[idx]),
                "confidence": float(probs[idx]),
                "proto_similarity": proto_sim,
                "attention_mean": float(attn_np.mean()),
            }
            if idx in dtw_meta:
                entry.update(dtw_meta[idx])
            predictions.append(entry)

        similarity_ok = top1 >= self.cfg.sim_threshold
        margin_ok = (top1 - top2) >= self.cfg.sim_margin if top2 > -1.0 else True
        proto_ok = True
        if predictions:
            proto_sim = predictions[0].get("proto_similarity")
            if proto_sim is not None:
                proto_ok = proto_sim >= (self.cfg.sim_threshold - 0.05)

        accepted = similarity_ok and margin_ok and proto_ok
        if not accepted:
            if not similarity_ok:
                reason = "low_similarity"
            elif not margin_ok:
                reason = "ambiguous_top2"
            else:
                reason = "prototype_disagreement"
        else:
            reason = "ok"

        return {
            "accepted": accepted,
            "reason": reason,
            "motion_mean": motion_mean,
            "top": predictions,
        }


# -----------------------------------------------------------------------------
# Real-time loop
# -----------------------------------------------------------------------------


def load_training_state(checkpoint_path: str, device: torch.device) -> tuple[Dict[str, object], Dict[str, object]]:
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state" in state:
        return state["model_state"], state.get("config", {})
    return state, {}


def build_model(cfg: RuntimeConfig, state_dict: Dict[str, object], device: torch.device) -> SignEmbeddingModel:
    model = SignEmbeddingModel(cfg)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_global_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats_blob = np.load(path)
    return stats_blob["feature_mean"].astype(np.float32), stats_blob["feature_std"].astype(np.float32)


def run_demo(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.embeddings):
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")
    if not os.path.exists(args.mapping):
        raise FileNotFoundError(f"Embeddings map not found: {args.mapping}")

    state_dict, ckpt_cfg = load_training_state(args.checkpoint, device)
    runtime_cfg = RuntimeConfig.from_training_cfg(ckpt_cfg)
    if args.disable_dtw:
        runtime_cfg.dtw_enabled = False
    if args.dtw_radius is not None:
        runtime_cfg.dtw_radius = args.dtw_radius

    global_stats_path = Path(args.global_stats or ckpt_cfg.get("global_stats_path", "./proc/global_stats.npz"))
    if not global_stats_path.exists():
        raise FileNotFoundError(f"Global stats not found: {global_stats_path}")
    prototypes_path = args.prototypes or os.path.join(Path(args.embeddings).parent, "prototypes.npz")
    if prototypes_path and not Path(prototypes_path).exists():
        prototypes_path = None

    global_mean, global_std = load_global_stats(global_stats_path)

    model = build_model(runtime_cfg, state_dict, device)
    engine = InferenceEngine(
        model=model,
        embeddings_path=args.embeddings,
        mapping_path=args.mapping,
        cfg=runtime_cfg,
        global_mean=global_mean,
        global_std=global_std,
        device=device,
        prototypes_path=prototypes_path,
    )

    mp_holistic = mp.solutions.holistic
    buffer: deque[np.ndarray] = deque(maxlen=runtime_cfg.seq_len * 2)
    cooldown_timer = 0.0
    last_result: Optional[Dict[str, object]] = None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}")

    print("\nControls: press \"q\" to quit, space to reset buffer.")
    print("Hold a sign for ~2 seconds to trigger retrieval.\n")

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            success, frame = cap.read()
            if not success:
                print("Stream ended or failed.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            feature = extract_feature_vector(results)
            if feature is not None:
                buffer.append(feature)

            now = time.time()
            if len(buffer) >= args.min_frames and (now - cooldown_timer) >= args.infer_every:
                seq = np.stack(buffer, axis=0)
                last_result = engine.predict(seq, topk=args.topk)
                cooldown_timer = now

            overlay_text(frame, last_result, len(buffer), args)

            cv2.imshow("GhSL real-time retrieval", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                buffer.clear()
                cooldown_timer = 0.0
                last_result = None

    cap.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Overlay helpers
# -----------------------------------------------------------------------------


def overlay_text(frame: np.ndarray, result: Optional[Dict[str, object]], buffer_len: int, args: argparse.Namespace) -> None:
    predictions = result.get("top", []) if result else []
    accepted = result.get("accepted", False) if result else False
    reason = result.get("reason", "idle") if result else "idle"
    motion_mean = result.get("motion_mean") if result else None

    h, w, _ = frame.shape
    panel_width = int(0.5 * w)
    panel_height = 30 * (max(len(predictions), 1) + 4)

    x0 = w - panel_width - 10
    y0 = 10
    cv2.rectangle(frame, (x0, y0), (x0 + panel_width, y0 + panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_width, y0 + panel_height), (255, 255, 255), 1)

    line_y = y0 + 25
    status_txt = "ACCEPTED" if accepted else f"REJECTED: {reason}" if result else "IDLE"
    status_color = (0, 220, 0) if accepted else (0, 180, 255) if result else (200, 200, 200)
    cv2.putText(frame, status_txt, (x0 + 10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    line_y += 25
    motion_txt = f"Motion mean: {motion_mean:.4f}" if isinstance(motion_mean, (int, float)) else "Motion mean: --"
    cv2.putText(frame, motion_txt, (x0 + 10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    line_y += 25
    cv2.putText(frame, f"Frames buffered: {buffer_len}", (x0 + 10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    line_y += 25
    cv2.putText(frame, "Top predictions:", (x0 + 10, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    for pred in predictions:
        line_y += 25
        sentence_txt = str(pred.get("sentence", ""))[:120]
        cv2.putText(
            frame,
            f"{pred.get('rank', '?')}. {sentence_txt}",
            (x0 + 10, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        line_y += 20
        confidence = float(pred.get("confidence", 0.0)) * 100.0
        proto = pred.get("proto_similarity")
        proto_txt = f", proto {proto:.3f}" if isinstance(proto, (int, float)) else ""
        cv2.putText(
            frame,
            f"confidence: {confidence:.1f}% | sim {pred.get('similarity', 0.0):.3f}{proto_txt}",
            (x0 + 20, line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def find_default_checkpoint() -> Optional[str]:
    runs_dir = Path("./runs")
    if not runs_dir.exists():
        return None
    for pattern in ("best_model_top1_*.pt", "best_model_r1_*.pt"):
        for candidate in sorted(runs_dir.glob(pattern)):
            return str(candidate)
    return None


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time GhSL sentence retrieval demo")
    parser.add_argument("--checkpoint", default=find_default_checkpoint() or "", help="Path to the trained model checkpoint (.pt)")
    parser.add_argument("--embeddings", default="./runs/embeddings.npy", help="Path to cached embeddings produced after training")
    parser.add_argument("--mapping", default="./runs/embeddings_map.csv", help="CSV mapping file for embeddings")
    parser.add_argument("--global-stats", default="", help="Path to global_stats.npz (defaults to checkpoint config)")
    parser.add_argument("--prototypes", default="", help="Optional prototypes.npz for prototype gating")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--min-frames", dest="min_frames", type=int, default=32, help="Minimum buffered frames before running inference")
    parser.add_argument("--infer-every", type=float, default=1.5, help="Seconds between inference calls")
    parser.add_argument("--topk", type=int, default=3, help="Number of predictions to display")
    parser.add_argument("--disable-dtw", action="store_true", help="Disable DTW re-ranking even if fastdtw is available")
    parser.add_argument("--dtw-radius", type=int, default=None, help="Override DTW radius when enabled")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    if not args.checkpoint:
        print("No checkpoint provided and none found in ./runs. Use --checkpoint to specify a file.")
        sys.exit(1)
    try:
        run_demo(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
