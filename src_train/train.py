import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk


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


if __name__ == '__main__':

    # Print current working directory
    current_directory = os.getcwd()
    print(f"当前工作目录: {current_directory}")

    print("Preparing training and validation data...")
    dataset_path = './wikitext'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading...")
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        dataset.save_to_disk(dataset_path)
        print("Dataset downloaded and saved to disk.")
    else:
        print("Dataset found. Loading from disk...")
        dataset = load_from_disk(dataset_path)
        print("Dataset loaded from disk.")

    print(f"目录内容: {os.listdir(dataset_path)}")

    # Set random seed
    seed = 29
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizer...")
    tokenizer_path = './gpt2_tokenizer'
    if not os.path.exists(tokenizer_path):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.save_pretrained(tokenizer_path)
        print("Tokenizer downloaded and saved.")
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        print("Tokenizer loaded from disk.")

    # Set pad_token
    if tokenizer.pad_token is None:
        #tokenizer.add_special_tokens({'pad_token': "<[PAD]>"})
        #tokenizer.pad_token=tokenizer.eos_token
        tokenizer.pad_token = '@'

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    #print(tokenizer.convert_tokens_to_ids('[PAD]'))
    #print(tokenizer.eos_token)

    print("Initializing model...")
    config = GPT2Config.from_json_file('./gpt2_config.json')
    model = GPT2LMHeadModel(config=config).to(device)

    #state_dict = torch.load('./gpt2_pth_5epochs/gpt2_model.pth')
    #model.load_state_dict(state_dict)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    print("Model initialized.")

    print("Preparing training and validation data...")
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']

    train_dataset = TextDataset(train_texts, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    print("Training data prepared.")

    val_dataset = TextDataset(val_texts, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=1)
    print("Validation data prepared.")

    vocab_size = len(tokenizer)
    total_epochs = 50

    writer = SummaryWriter(log_dir='./logs/gpt2_modified')

    print('start training!')

    for epoch in range(total_epochs):
        print(f"Starting epoch {epoch + 1}/{total_epochs}...")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"epoch {epoch + 1}/{total_epochs}")
        for i, batch in enumerate(progress_bar):
            # Move the batch's tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']

            # Create labels by shifting inputs to the right
            labels = inputs[:, 1:].clone()
            labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss

            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=attention_mask)

            # Adjust outputs to match labels
            outputs = outputs.logits[:, :-1, :]  # Shift outputs to match the labels
            loss = loss_fn(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))  # Calculate loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataloader) + i)

        scheduler.step()
        print(f"Epoch {epoch + 1} completed. Loss: {total_loss / len(train_dataloader)}")
        torch.save(model.state_dict(), './gpt2_pth/gpt2_model_modified.pth')


    writer.close()
    print("Training completed.")
