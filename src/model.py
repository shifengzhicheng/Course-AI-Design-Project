import os
import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk

# 加载 GPT-2 配置
config = GPT2Config(
    vocab_size=50258,  # GPT-2 的词汇表大小
    n_positions=1024,  # 设置位置编码的最大长度
    n_embd=256,        # 隐藏层大小
    n_layer=4,        # Transformer 层数
    n_head=4          # 注意力头的数量
)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

class NanoGPT(nn.Module):
    def __init__(self, config):
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
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.positional_encoding = nn.Parameter(torch.zeros(1, config.n_positions, config.n_embd))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)  # 初始化位置编码
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=0.1,
                activation='gelu'
            )
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.fc_out = nn.Linear(config.n_embd, config.vocab_size)
    def forward(self, input_ids):
        input_ids = input_ids.long()
        embeddings = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        embeddings = embeddings.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        seq_length = embeddings.size(0)
        mask = generate_square_subsequent_mask(seq_length).to(embeddings.device)

        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_mask=mask)

        hidden_states = self.ln_f(hidden_states)
        logits = self.fc_out(hidden_states.transpose(0, 1))
        return logits  # 返回形状为 (batch_size, seq_len, vocab_size)
    