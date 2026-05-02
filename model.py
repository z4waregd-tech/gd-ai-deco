import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GDEncoderDecoderTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for GD decoration.

    Encoder:   reads the FULL gameplay sequence + theme tag
    Decoder:   autoregressively generates the decoration sequence,
               attending to the encoder output at every step (cross-attention)

    This is fundamentally better than the old decoder-only (GPT) approach because
    the model can see the COMPLETE level layout before placing a single deco object,
    instead of only seeing a sliding 1024-token window.
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Shared embedding — encoder and decoder use the same token vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_head = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.zeros_(self.output_head.bias)
        nn.init.uniform_(self.output_head.weight, -0.1, 0.1)

    def _embed(self, ids):
        """Embed + scale + positional encode. ids: [batch, seq]"""
        x = self.embedding(ids) * math.sqrt(self.d_model)
        # PositionalEncoding expects [seq, batch, d_model]
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        return x.transpose(0, 1)   # back to [batch, seq, d_model]

    def _causal_mask(self, sz, device):
        """Upper-triangular -inf mask so decoder can't peek at future deco tokens."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        src: [batch, src_seq]   — gameplay tokens (encoder input)
        tgt: [batch, tgt_seq]   — decoration tokens shifted right (decoder input)
        Returns logits: [batch, tgt_seq, vocab_size]
        """
        src_emb = self._embed(src)
        tgt_emb = self._embed(tgt)

        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_head(out)

    def encode(self, src, src_key_padding_mask=None):
        """Run only the encoder — used during generation."""
        src_emb = self._embed(src)
        # Access internal encoder
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode_step(self, tgt, memory, src_key_padding_mask=None):
        """
        Run only the decoder for one autoregressive step.
        tgt:    [1, tgt_len]
        memory: [1, src_len, d_model]  (encoder output, cached)
        """
        tgt_emb = self._embed(tgt)
        tgt_mask = self._causal_mask(tgt.size(1), tgt.device)
        out = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_head(out)


if __name__ == "__main__":
    print("Testing GDEncoderDecoderTransformer...")
    vocab_size = 15000
    model = GDEncoderDecoderTransformer(vocab_size=vocab_size)

    src = torch.randint(1, vocab_size, (4, 512))   # gameplay (encoder)
    tgt = torch.randint(1, vocab_size, (4, 256))   # deco shifted right (decoder)

    logits = model(src, tgt)
    print(f"src shape:    {src.shape}")
    print(f"tgt shape:    {tgt.shape}")
    print(f"logits shape: {logits.shape}")  # [4, 256, vocab_size]
    print("OK!")
