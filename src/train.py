import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, load_from_disk
from model import NanoGPT, config
import torch
import os
import torch.nn as nn
from transformers import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 准备数据
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

def top_k_logits(logits, k):
    if k == 0:
        # 不进行截断
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50):
    """
    Generates text using a pre-trained language model.

    Args:
        model (torch.nn.Module): The pre-trained language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
        prompt (str): The initial text prompt to start the generation.
        max_length (int, optional): The maximum length of the generated text. Defaults to 50.
        temperature (float, optional): The temperature to use for sampling. Higher values result in more random samples. Defaults to 1.0.
        top_k (int, optional): The number of top tokens to consider for sampling. Defaults to 50.

    Returns:
        str: The generated text.

    Example:
        >>> generated_text = generate_text(model, tokenizer, "Once upon a time", max_length=100)
        >>> print(generated_text)
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :] / temperature
        filtered_logits = top_k_logits(next_token_logits, top_k)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# 创建一个 collate 函数
def collate_fn(batch):
    input_ids = [torch.tensor(example['input_ids']) for example in batch]
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return {
        'input_ids': input_ids_padded.to(device),
    }
        
if __name__ == '__main__':
    # Print current working directory
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

    # Set random seed for reproducibility
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

    # Set pad_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    
    # Update config's vocab_size
    config.vocab_size = len(tokenizer)
    print("Preparing training and validation data...")

    # Tokenize the dataset
    tokenized_datasets = dataset['train'].map(tokenize_function, batched=True)

    # Remove 'text' column
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    print("Initializing model...")
    model = NanoGPT(config).to(device)

    # Set training parameters
    train_args = {
        "batch_size": 32,
        "learning_rate": 5e-5,
        "num_epochs": 5,
    }

    # Create DataLoader
    train_loader = DataLoader(tokenized_datasets, batch_size=train_args['batch_size'], shuffle=True, collate_fn=collate_fn)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=train_args['learning_rate'])
    print("Model initialized.")

    writer = SummaryWriter(log_dir='./logs/nanoGPT')

    total_epochs = 50
    print('')
    print('start training!')
    print('')

    for epoch in range(train_args['num_epochs']):
        total_loss = 0
        print(f"Starting epoch {epoch + 1}")
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")):
            input_ids = batch['input_ids']

            if step == 0:
                print(
                    f"input_ids range: max={input_ids.max().item()}, min={input_ids.min().item()}, vocab_size={config.vocab_size}")

            if input_ids.max().item() >= config.vocab_size or input_ids.min().item() < 0:
                raise ValueError("input_ids 包含超出词汇表范围的值！")

            # Forward pass
            outputs = model(input_ids)

            # Shift labels to the right
            labels = input_ids[:, 1:].clone()
            labels[labels == tokenizer.pad_token_id] = -100  # Use -100 to ignore these positions in loss calculation

            # Align outputs with labels
            outputs = outputs[:, :-1, :]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print loss every 5 steps
            if step % 5 == 0:
                print(f"Step {step}, Loss: {loss.item()}")
                writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch: {epoch + 1}, Average Loss: {avg_loss}")
        torch.save(model.state_dict(), './nanoGPT_model.pth')
        tokenizer.save_pretrained("./")
        writer.add_scalar('Average Loss', avg_loss, epoch)
        
    writer.close()
    print("Training completed.")

    # Example: Generate text
    prompt = "Once upon a time"
    generated_text = generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50)
    print(f"Generated Text:\n{generated_text}")
