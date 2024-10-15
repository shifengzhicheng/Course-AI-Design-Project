import torch
from transformers import GPT2Tokenizer
import argparse
import torch.nn as nn
from model import NanoGPT

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--input_text', type=str, required=True, help='Input text for generating')
parser.add_argument('--max_length', type=int, default=50, help='Maximum length of generated text')
parser.add_argument('--top_k', type=int, default=50, help='Top-K sampling')

args = parser.parse_args()

# 加载 tokenizer 和 NanoGPT 模型
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2_tokenizer')
vocab_size = len(tokenizer)
model = NanoGPT(vocab_size, 256, 4, 4)
#model.load_state_dict(torch.load('./nanoGPT_model.pth'))
model.eval()

# 获取输入文本并进行编码
input_text = args.input_text
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 禁用梯度计算
with torch.no_grad():
    for _ in range(args.max_length - len(input_ids[0])):
        # 前向传递模型
        output = model(input_ids)

        # 取出输出的最后一个 token 的分布
        logits = output[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)

        # 使用 Top-K 采样
        top_k = args.top_k
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k)

        # 在 top_k 概率分布上进行采样
        next_token_index = torch.multinomial(top_k_probs, 1)  # 采样，返回一个标量
        next_token = top_k_indices[0, next_token_index.item()]  # 取出对应的 token 索引
        next_token = next_token.unsqueeze(0).unsqueeze(0)  # 变为 (1, 1)

        # 将生成的 token 添加到输入中
        input_ids = torch.cat((input_ids, next_token), dim=1)

        # 如果生成结束符就停止
        if next_token.item() == tokenizer.eos_token_id:
            break

    # 解码生成的所有 token 为文本
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
