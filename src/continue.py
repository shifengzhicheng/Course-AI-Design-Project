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
from model import NanoGPT
from model import TextDataset

if __name__ == '__main__':

    # print current working directory
    current_directory = os.getcwd()
    print(f"当前工作目录: {current_directory}")
    
    print("Preparing training and validation data...")
    # dataset_path = './bookcorpus'
    dataset_path = './wikitext'
    if not os.path.exists(dataset_path):
        print("Dataset not found. Downloading...")
        if dataset_path == './bookcorpus':
            dataset = load_dataset('bookcorpus', 'plain_text')
        if dataset_path == './wikitext':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        dataset.save_to_disk(dataset_path)
        print("Dataset downloaded and saved to disk.")
    else:
        print("Dataset found. Loading from disk...")
        dataset = load_from_disk(dataset_path)
        print("Dataset loaded from disk.")

    print(f"目录内容: {os.listdir(dataset_path)}")

    # set random seed
    ###################################################
    seed = 29
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ###################################################

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
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

    # set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Preparing training and validation data...")
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']

    train_dataset = TextDataset(train_texts, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("Training data prepared.")

    val_dataset = TextDataset(val_texts, tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    print("Validation data prepared.")

    vocab_size = len(tokenizer)
    n_embd = 256  # embedded
    n_layer = 8  # layers
    n_head = 8  # attention head
    print("Initializing model...")
    model = NanoGPT(vocab_size, n_embd, n_layer, n_head).to(device)

    # Load previous model weights if available
    model_path = './nanoGPT_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded previous model weights.")
    else:
        print("No previous model weights found. Training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    print("Model initialized.")

    writer = SummaryWriter(log_dir='./logs/nanoGPT')

    total_epochs = 50
    print('')
    print('start training!')
    print('')
    for epoch in range(total_epochs):
        print(f"Starting epoch {epoch + 1}/{total_epochs}...")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"epoch {epoch + 1}/{total_epochs}")
        for i, batch in enumerate(progress_bar):
            # Move the batch's tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = batch['input_ids']  # Adjust if you need other elements from the batch
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, vocab_size), batch['input_ids'].view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataloader) + i)
        scheduler.step()
        print(f"Epoch {epoch + 1} completed. Loss: {total_loss / len(train_dataloader)}")
        torch.save(model.state_dict(), './nanoGPT_model.pth')

        if (epoch + 1) % 5 == 0:
            print(f"Evaluating model at epoch {epoch + 1}...")
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    # Move the batch's tensors to the device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    inputs = batch['input_ids']  # Adjust if you need other elements from the batch
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs.view(-1, vocab_size), batch.view(-1))
                    total_val_loss += loss.item()

            print(f"Epoch {epoch + 1}, Validation Loss: {total_val_loss / len(val_dataloader)}")
            writer.add_scalar('Val Loss', total_val_loss / len(val_dataloader), epoch)

    writer.close()
    print("Training completed.")