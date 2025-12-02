import math
from pathlib import Path

import torch
import torch.nn as nn


class FrameProjector(nn.Module):
    """Project per-frame features to a lower-dimensional representation."""

    def __init__(self, in_dim: int, proj_dim: int = 160) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F)
        out = self.norm(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.act(self.fc2(out))
        return out


class TemporalEncoder(nn.Module):
    """Encode temporal sequence into a fixed embedding with attention pooling."""

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
        self.context_proj = nn.Linear(hidden * 2, embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, return_sequence: bool = False):  # (B, T, C)
        seq_out, _ = self.gru(x)  # (B, T, 2*hidden)

        keys = torch.tanh(self.attn_key(seq_out))  # (B, T, hidden)
        attn_logits = self.attn_score(keys)  # (B, T, H)

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

        if return_sequence:
            context = self.context_proj(seq_out)
            return embedding, weights.mean(dim=2), context

        return embedding, weights.mean(dim=2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SignTranslationModel(nn.Module):
    """Seq2Seq model: Visual Encoder + Text Decoder."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        proj_dim: int = 160,
        hidden: int = 160,
        n_layers: int = 2,
        embed_dim: int = 256,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()
        self.projector = FrameProjector(input_dim, proj_dim)
        self.encoder = TemporalEncoder(proj_dim, hidden, n_layers, embed_dim, attn_heads)
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=attn_heads, dim_feedforward=embed_dim * 4, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.generator = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def encode_visual(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.projector(x)
        embedding, _ = self.encoder(proj)
        return embedding

    def encode_for_inference(self, x: torch.Tensor):
        proj = self.projector(x)
        embedding, attn_weights, context = self.encoder(proj, return_sequence=True)
        return embedding, proj, attn_weights

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask=None, tgt_pad_mask=None):
        proj = self.projector(src)
        embedding, attn_weights, context = self.encoder(proj, return_sequence=True)
        memory = context.permute(1, 0, 2)
        tgt_emb = self.tgt_embed(tgt).permute(1, 0, 2)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        logits = self.generator(output.permute(1, 0, 2))
        return logits, embedding

    @torch.no_grad()
    def project_frames(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)
