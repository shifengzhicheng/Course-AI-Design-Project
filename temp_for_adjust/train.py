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

dataset = load_from_disk('./wikitext')

tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_tokenizer')
train_texts = dataset['train']['text']
train_dataset = TextDataset(train_texts, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_texts = dataset['validation']['text']
val_dataset = TextDataset(val_texts, tokenizer)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

vocab_size = len(tokenizer)
n_embd = 256  # embedded
n_layer = 4  # layers
n_head = 4  # attention head
model = NanoGPT(vocab_size, n_embd, n_layer, n_head).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()

writer = SummaryWriter(log_dir='./logs/nanoGPT')

total_epochs = 50
print('')
print('start training!')
print('')
for epoch in range(total_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader,desc=f"epoch{epoch+1}/{total_epochs}")
    for i, batch in enumerate(progress_bar):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs.view(-1, vocab_size), batch.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss':loss.item()})
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_dataloader) + i)
    scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")
    torch.save(model.state_dict(), './nanoGPT_model.pth')

    if (epoch + 1) % 5 == 0:
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = loss_fn(outputs.view(-1,vocab_size),batch.view(-1))
                total_val_loss += loss.item()

        print(f"epoch{epoch+1},val_loss:{total_val_loss / len(val_dataloader)}")
        writer.add_scalar('Val Loss', total_val_loss / len(val_dataloader), epoch)

writer.close()



