import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

def top_k_logits(logits, k):
    if k == 0:
        # 不进行截断
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1].unsqueeze(1)
        return torch.where(logits < min_values, torch.full_like(logits, -float('Inf')), logits)

def generate_text(model, tokenizer, prompt, max_length, temperature, top_k):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    count = 0
    for i in range(max_length):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :] / temperature
        filtered_logits = top_k_logits(next_token_logits, top_k)
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        #print(next_token.item())
        if next_token.item() == 764:
            count += 1
            if i >= 55:
                count += 1
        if next_token.item() == tokenizer.eos_token_id or count >=2:
            break
    sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    #sentence = input_ids[0]
    return sentence

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--input_text', type=str, required=True, help='Input text for generating')
    parser.add_argument('--max_length', type=int, default=80, help='Maximum length of generated text')
    parser.add_argument('--top_k', type=int, default=20, help='Top-K sampling')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')

    args = parser.parse_args()

    # 加载 tokenizer 和 NanoGPT 模型
    tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_tokenizer')
    config = GPT2Config.from_json_file('./gpt2_config.json')
    model = GPT2LMHeadModel(config=config)
    #state_dict = torch.load('./gpt2_pth/gpt2_model_modified.pth')
    state_dict = torch.load('./gpt2_pth_5epochs/gpt2_model.pth')
    model.load_state_dict(state_dict)
    model.eval()

    # 获取输入文本并生成文本
    generated_text = generate_text(model, tokenizer, args.input_text, max_length=args.max_length, temperature=args.temperature, top_k=args.top_k)
    print(f"Generated: {generated_text}")
    #print(tokenizer.encode('S', return_tensors='pt'))
    # Example usage:
    # python generate.py -t "Once upon a time" --max_length 100 --top_k 40 --temperature 0.7