import torch
import torch.nn as nn
import math
import json
from pathlib import Path

class FrameProjector(nn.Module):
    def __init__(self, in_dim, proj_dim=160):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.norm(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.act(self.fc2(out))
        return out

class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, hidden=160, n_layers=2, embed_dim=256, attn_heads=4):
        super().__init__()
        self.gru = nn.GRU(
            in_dim,
            hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if n_layers > 1 else 0.0
        )
        self.attn_heads = attn_heads
        self.attn_key = nn.Linear(hidden * 2, hidden)
        self.attn_score = nn.Linear(hidden, attn_heads)
        self.attn_proj = nn.Linear(hidden * 2, embed_dim)
        self.context_proj = nn.Linear(hidden * 2, embed_dim)
        self.post_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, return_sequence=False):
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

        if return_sequence:
            context = self.context_proj(seq_out)
            return embedding, weights.mean(dim=2), context
        return embedding, weights.mean(dim=2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SignTranslationModel(nn.Module):
    def __init__(self, input_dim, vocab_size, proj_dim=160, hidden=160, n_layers=2, embed_dim=256, attn_heads=4):
        super().__init__()
        self.projector = FrameProjector(input_dim, proj_dim)
        self.encoder = TemporalEncoder(proj_dim, hidden, n_layers, embed_dim, attn_heads)
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=attn_heads, dim_feedforward=embed_dim*4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.generator = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

    def forward(self, src, tgt, tgt_mask=None, tgt_pad_mask=None):
        proj = self.projector(src)
        embedding, attn_weights, context = self.encoder(proj, return_sequence=True)
        memory = context.permute(1, 0, 2)
        tgt_emb = self.tgt_embed(tgt).permute(1, 0, 2)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        logits = self.generator(output.permute(1, 0, 2))
        return logits, embedding

class SimpleTokenizer:
    def __init__(self, vocab_path: Path):
        with open(vocab_path, 'r') as fp:
            data = json.load(fp)
        token2id = data['token2id']
        self.token2id = token2id
        self.id2token = {int(v): k for k, v in token2id.items()}
        self.vocab_size = len(self.token2id)
        self.pad_token_id = self.token2id.get('<pad>', 0)
        self.sos_token_id = self.token2id.get('<sos>', 1)
        self.eos_token_id = self.token2id.get('<eos>', 2)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        words = []
        for idx in ids:
            token = self.id2token.get(int(idx), '<unk>')
            if skip_special_tokens and token in {'<pad>', '<sos>', '<eos>'}:
                continue
            words.append(token)
        return ' '.join(words)
