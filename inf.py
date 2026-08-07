# python inf.py --checkpoint "./checkpoints/checkpoint-80/model.safetensors"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MyModel
import argparse
import os
from safetensors.torch import load_file

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
            # 构造 key_padding_mask（无填充，全 False）
            L = generated.size(1)
            key_padding_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
            
            # 前向
            logits = model(generated, key_padding_mask=key_padding_mask)  # (1, L, vocab_size)
            next_token_logits = logits[:, -1, :] / temperature   # 取最后一个位置的 logits
            
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
            
            # 可选：遇到 EOS 终止（Qwen tokenizer 有 eos_token）
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码
    output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    return output_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径（.pt 或 .bin）")
    parser.add_argument("--prompt", type=str, default="我，", help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度（0 为贪心）")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k 采样（None 则关闭）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    device = args.device
    if not torch.cuda.is_available():
        device = "cpu"

    # 加载 tokenizer（与训练一致）
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

    # 重新构建模型（参数必须与训练时完全一致）
    model = MyModel(
        vocab_size=tokenizer.vocab_size,
        num_layers=24,
        d_model=2048,
        num_heads=16,
        num_kv_heads=8,
        dropout=0.1,          # 推理时 dropout 不影响，但为了加载准确保持相同
        max_seq_len=2048
    ).to(device)

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
        device=device
    )
    print("\n=== Generated Text ===\n")
    print(output)

if __name__ == "__main__":
    main()