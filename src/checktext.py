import torch
import torch.nn as nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk
from model import NanoGPT
from dataset import TextDataset

if __name__ == '__main__':

    dataset = load_from_disk('./wikitext')

    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_tokenizer')
    train_texts = dataset['train']['text']
    train_dataset = TextDataset(train_texts, tokenizer)
    print(len(train_dataset))

