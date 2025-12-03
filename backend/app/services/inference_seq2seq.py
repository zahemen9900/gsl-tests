from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .inference_model import SignTranslationModel

CFG = {
    "seq_len": 64,
    "input_dim": 540,
    "proj_dim": 160,
    "embed_dim": 256,
    "attn_heads": 4,
    "encoder_layers": 4,
    "encoder_ff_dim": 512,
    "encoder_dropout": 0.1,
    "decoder_max_len": 64,
}


class SimpleTokenizer:
    def __init__(self, vocab_path: Path) -> None:
        with open(vocab_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        token2id = data["token2id"]
        self.token2id = token2id
        self.id2token = {int(v): k for k, v in token2id.items()}
        self.vocab_size = len(self.token2id)
        self.pad_token_id = self.token2id.get("<pad>", 0)
        self.sos_token_id = self.token2id.get("<sos>", 1)
        self.eos_token_id = self.token2id.get("<eos>", 2)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = text.lower().replace(",", "").replace(".", "").split()
        ids = [self.token2id.get(tok, self.token2id.get("<unk>", 3)) for tok in tokens]
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        for idx in ids:
            token = self.id2token.get(int(idx), "<unk>")
            if skip_special_tokens and token in {"<pad>", "<sos>", "<eos>"}:
                continue
            words.append(token)
        return " ".join(words)


class Seq2SeqInferenceService:
    """Lightweight streaming decoder exposed via websockets."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.runs_dir = self.root_dir / "runs"
        self.proc_dir = self.root_dir / "proc"
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.global_mean, self.global_std = self._load_stats()
        self.buffer: List[List[float]] = []
        self.buffer_max_len = CFG["seq_len"]

    def _load_tokenizer(self) -> SimpleTokenizer:
        vocab_path = self.runs_dir / "tokenizer_vocab.json"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Tokenizer vocab not found at {vocab_path}")
        return SimpleTokenizer(vocab_path)

    def _load_model(self) -> SignTranslationModel:
        candidates = sorted(self.runs_dir.glob("best_model_top1_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found in {self.runs_dir}")
        ckpt_path = candidates[-1]
        model = SignTranslationModel(
            input_dim=CFG["input_dim"],
            vocab_size=self.tokenizer.vocab_size,
            proj_dim=CFG["proj_dim"],
            embed_dim=CFG["embed_dim"],
            attn_heads=CFG["attn_heads"],
            encoder_layers=CFG["encoder_layers"],
            encoder_ff_dim=CFG["encoder_ff_dim"],
            encoder_dropout=CFG["encoder_dropout"],
            max_seq_len=CFG["seq_len"]
        ).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_stats(self) -> tuple[np.ndarray, np.ndarray]:
        stats_path = self.proc_dir / "global_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats not found at {stats_path}")
        blob = np.load(stats_path)
        return blob["feature_mean"].astype(np.float32), blob["feature_std"].astype(np.float32)

    def normalize(self, seq: np.ndarray) -> np.ndarray:
        return (seq - self.global_mean) / (self.global_std + 1e-6)

    def process_frame(self, frame: List[float]) -> Optional[str]:
        if len(frame) != CFG["input_dim"]:
            return None
        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_max_len:
            self.buffer.pop(0)
        if len(self.buffer) >= 16 and len(self.buffer) % 5 == 0:
            return self._decode()
        return None

    def _decode(self) -> Optional[str]:
        seq = np.array(self.buffer, dtype=np.float32)
        seq = self.normalize(seq)
        if seq.shape[0] < CFG["seq_len"]:
            pad = np.zeros((CFG["seq_len"] - seq.shape[0], CFG["input_dim"]), dtype=np.float32)
            seq = np.vstack([seq, pad])
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ys = torch.full((1, 1), self.tokenizer.sos_token_id, dtype=torch.long, device=self.device)
            proj = self.model.projector(tensor)
            _, _, context = self.model.encoder(proj, return_sequence=True)
            memory = context.permute(1, 0, 2)
            for _ in range(CFG["decoder_max_len"]):
                tgt_emb = self.model.tgt_embed(ys).permute(1, 0, 2)
                tgt_emb = self.model.pos_encoder(tgt_emb)
                sz = ys.size(1)
                mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0)).to(
                    self.device
                )
                out = self.model.decoder(tgt_emb, memory, tgt_mask=mask)
                logits = self.model.generator(out.permute(1, 0, 2))
                next_token = logits[:, -1, :].argmax(dim=-1)
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            return self.tokenizer.decode(ys[0].tolist(), skip_special_tokens=True)

    def reset(self) -> None:
        self.buffer = []
