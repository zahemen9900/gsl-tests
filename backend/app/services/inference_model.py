import math
import torch
import torch.nn as nn


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


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.size(1)
        return self.dropout(x + self.pe[:, :length])


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim=256,
        n_layers=4,
        attn_heads=4,
        ff_dim=512,
        dropout=0.1,
        max_len=256
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.pos_encoder = TemporalPositionalEncoding(embed_dim, dropout=dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attn_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, return_sequence=False):
        seq = self.input_proj(x)
        seq = self.pos_encoder(seq)
        seq = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        pooled = torch.mean(seq, dim=1)
        pooled = nn.functional.normalize(self.norm(pooled), dim=-1)
        attn_proxy = torch.ones(x.size(0), x.size(1), device=x.device)
        if return_sequence:
            return pooled, attn_proxy, seq
        return pooled, attn_proxy


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
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class SignTranslationModel(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        proj_dim=160,
        embed_dim=256,
        attn_heads=4,
        encoder_layers=4,
        encoder_ff_dim=512,
        encoder_dropout=0.1,
        max_seq_len=64
    ):
        super().__init__()
        self.projector = FrameProjector(input_dim, proj_dim)
        self.encoder = TemporalTransformerEncoder(
            proj_dim,
            embed_dim=embed_dim,
            n_layers=encoder_layers,
            attn_heads=attn_heads,
            ff_dim=encoder_ff_dim,
            dropout=encoder_dropout,
            max_len=max_seq_len
        )
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=attn_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.generator = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.mask_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )

    def encode_visual(self, x):
        proj = self.projector(x)
        embedding, _ = self.encoder(proj)
        return embedding

    def encode_for_inference(self, x):
        proj = self.projector(x)
        embedding, attn_weights, context = self.encoder(proj, return_sequence=True)
        return embedding, proj, attn_weights

    def forward(self, src, tgt, tgt_mask=None, tgt_pad_mask=None):
        proj = self.projector(src)
        embedding, _, context = self.encoder(proj, return_sequence=True)
        memory = context.permute(1, 0, 2)
        tgt_emb = self.tgt_embed(tgt).permute(1, 0, 2)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        logits = self.generator(output.permute(1, 0, 2))
        return logits, embedding

    @torch.no_grad()
    def project_frames(self, x):
        return self.projector(x)

    def masked_frame_loss(self, src, mask_ratio=0.15):
        proj = self.projector(src)
        bsz, seq_len, _ = proj.shape
        mask = (torch.rand(bsz, seq_len, device=proj.device) < mask_ratio)
        if not mask.any():
            rand_b = torch.randint(0, bsz, (1,), device=proj.device)
            rand_t = torch.randint(0, seq_len, (1,), device=proj.device)
            mask[rand_b, rand_t] = True
        masked_proj = proj.clone()
        masked_proj[mask] = self.mask_token.expand_as(proj)[mask]
        _, _, context = self.encoder(masked_proj, return_sequence=True)
        recon = self.mask_decoder(context)
        loss = nn.functional.smooth_l1_loss(recon[mask], proj[mask])
        return loss
