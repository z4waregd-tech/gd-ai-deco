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
        # x shape: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GDAutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        
        # We use a standard Transformer Encoder but with a causal mask,
        # which effectively makes it function exactly like a GPT decoder!
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.linear_head = nn.Linear(d_model, vocab_size)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear_head.bias.data.zero_()
        self.linear_head.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        # This prevents the AI from "looking into the future" at tokens it hasn't generated yet.
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x):
        """
        Input shape: [batch_size, seq_len]
        Output shape: [batch_size, seq_len, vocab_size]
        """
        # Create a causal mask for the sequence
        seq_len = x.size(1)
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 1. Embed the tokens
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # We transpose to [seq_len, batch_size, embedding_dim] for the PositionalEncoding, 
        # then transpose back
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # 2. Pass through Transformer with causal mask
        # Note: PyTorch's TransformerEncoder expects is_causal=True or mask.
        output = self.transformer_encoder(x, mask=causal_mask, is_causal=True)
        
        # 3. Predict the next token logits
        logits = self.linear_head(output)
        
        return logits

if __name__ == "__main__":
    # Quick Test to ensure the model architecture complies
    print("Testing GDPR Autoregressive Transformer...")
    vocab_size = 861  # matching our dataset
    model = GDAutoregressiveTransformer(vocab_size=vocab_size)
    
    dummy_input = torch.randint(0, vocab_size, (4, 1024)) # [Batch, SeqLen]
    print(f"Dummy Input shape: {dummy_input.shape}")
    
    logits = model(dummy_input)
    print(f"Logits output shape: {logits.shape}") # Should be [4, 1024, 861]
    print("Model initialized and passed dummy data successfully!")
