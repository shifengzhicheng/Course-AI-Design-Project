import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.input_ids = []
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - block_size + 1, block_size):
                self.input_ids.append(tokens[i:i + block_size])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx], dtype=torch.long)

