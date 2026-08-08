# python inf.py --checkpoint "./checkpoints/checkpoint-80/model.safetensors"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import os
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM
from train import model_config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
model = Qwen2ForCausalLM(model_config).to(DEVICE)
model.config.use_cache = True

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=1.0,
    top_k=None,
    device="cuda"
):
    """
    自回归生成文本（无 KV cache，简单实现）
    """
    model.eval()
    # 编码 prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)
    
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            L = generated.size(1)
            # 前向
            logits = model(generated).logits   # (1, L, vocab_size)
            #next_token_logits = logits[:, -1, :] / temperature   # 取最后一个位置的 logits
            if temperature > 0:
                next_token_logits = logits[:, -1, :] / temperature
            else:
                next_token_logits = logits[:, -1, :]   # 不做缩放，直接 argmax
            
            # 可选 top-k 采样
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                # 将非 top-k 位置的 logits 设为 -inf
                mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                mask.scatter_(1, top_k_indices, False)
                next_token_logits = next_token_logits.masked_fill(mask, float('-inf'))
            
            # 采样或贪心
            if temperature > 0:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (1, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(1) >=model.config.max_position_embeddings:
                break 
            # 可选：遇到 EOS 终止（Qwen tokenizer 有 eos_token）
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径（.pt 或 .bin）")
    parser.add_argument("--prompt", type=str, default="我，", help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度（0 为贪心）")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k 采样（None 则关闭）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    # 加载训练好的权重
    if os.path.isfile(args.checkpoint):
        state_dict = load_file(args.checkpoint)
        #model.load_state_dict(state_dict, strict=False)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # 生成
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=DEVICE
    )
    print(output)