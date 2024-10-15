import torch
import torch.nn as nn

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super(NanoGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)  # Embedding layer to convert token IDs to embeddings
        self.transformer = nn.Transformer(
            d_model=n_embd,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        # Add embedding layer to map token IDs to embeddings
        x = self.embedding(x)  # x shape: (batch_size, sequence_length, n_embd)

        # Transpose to (sequence_length, batch_size, d_model) as expected by nn.Transformer
        x = x.transpose(0, 1)

        # Pass through Transformer layers
        x = self.transformer(x, x)  # Transformer expects src and tgt, both are x here

        # Transpose back to (batch_size, sequence_length, n_embd)
        x = x.transpose(0, 1)

        # Apply LayerNorm
        x = self.ln_f(x)

        # Output layer: map to vocab size
        return self.head(x)

'''model = NanoGPT(1000,256,4,8)
input = torch.randint(0,1000,(32,50))
output = model(input)
print(output.shape)'''
