import os
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        """
        Initializes the NanoGPT model.

        Args:
            vocab_size (int): The size of the vocabulary.
            n_embd (int): The dimensionality of the embeddings.
            n_layer (int): The number of layers in the transformer.
            n_head (int): The number of attention heads in the transformer.

        Attributes:
            embedding (nn.Embedding): Embedding layer to convert token IDs to embeddings.
            transformer (nn.Transformer): Transformer model for sequence-to-sequence tasks.
            ln_f (nn.LayerNorm): Layer normalization applied to the final output.
            head (nn.Linear): Linear layer to project the transformer output to vocabulary size.
        """
        super(NanoGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)  # Embedding layer to convert token IDs to embeddings
        self.transformer = nn.Transformer(
            d_model=n_embd,
            nhead=n_head,
            num_encoder_layers=n_layer,
            num_decoder_layers=n_layer,
            batch_first=True
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    def forward(self, x):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size).

        The forward pass includes the following steps:
        1. Embedding layer to map token IDs to embeddings.
        2. Transpose the tensor to shape (sequence_length, batch_size, d_model) for the Transformer.
        3. Pass the tensor through Transformer layers.
        4. Transpose the tensor back to shape (batch_size, sequence_length, n_embd).
        5. Apply LayerNorm.
        6. Output layer to map to vocabulary size.
        """
        x = self.embedding(x)  # x shape: (batch_size, sequence_length, n_embd)
        x = x.transpose(0, 1)
        x = self.transformer(x, x)  # Transformer expects src and tgt, both are x here
        x = x.transpose(0, 1)
        x = self.ln_f(x)
        return self.head(x)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
    def __len__(self):
        return len(self.encodings['input_ids'])
    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        return item