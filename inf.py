# inference.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import MyModel          # 与训练时相同的模型定义
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ========== 1. 加载 tokenizer（必须与训练时一致） ==========
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)

# ========== 2. 加载模型 ==========
def load_model(checkpoint_dir, device=DEVICE):
    """
    从 HuggingFace Trainer 保存的 checkpoint 目录加载模型。
    参数与训练时保持一致：
        vocab_size=tokenizer.vocab_size,
        num_layers=24,
        d_model=2048,
        num_heads=16,
        num_kv_heads=8,
        max_seq_len=8192
    """
    model = MyModel(
        vocab_size=tokenizer.vocab_size,
        num_layers=24,
        d_model=2048,
        num_heads=16,
        num_kv_heads=8,
        dropout=0.1,          # 推理时 dropout 不起作用，但为了参数匹配
        max_seq_len=8192
    )
    # 尝试加载 .bin 文件，若不存在则加载 .safetensors
    bin_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
    safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')
    if os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location='cpu')
    elif os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ========== 3. 生成函数（top‑p 采样，支持温度调节） ==========
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    stop_on_eos=True
):
    """
    自回归生成文本。
    每次重新构造因果掩码（未使用 KV cache），适合短文本演示。
    """
    # 编码 prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(DEVICE)
    
    for _ in range(max_new_tokens):
        if input_ids.size(1) >= model.max_seq_len:
            break
        
        L = input_ids.size(1)
        # 构造因果掩码 (1, 1, L, L)
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.float32), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(DEVICE)
        # 无 padding，全部为 False
        key_padding_mask = torch.zeros(1, L, dtype=torch.bool).to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_ids, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        
        # 取最后一个位置的 logits
        logits_last = logits[0, -1, :] / temperature
        
        # top‑p (nucleus) 采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_last, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits_last[indices_to_remove] = float('-inf')
        
        probs = F.softmax(logits_last, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        # 如果遇到 EOS 且要求停止，则退出
        if stop_on_eos and tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

# ========== 4. 主程序示例 ==========
if __name__ == "__main__":
    # 请根据实际 checkpoint 路径修改
    checkpoint_dir = "./checkpoints"   # 如果保存了最佳模型，可能是 "./checkpoints/checkpoint-xxx"
    model = load_model(checkpoint_dir)
    
    # 你可以在这里修改 prompt，或从命令行传入
    prompt = "Once upon a time"
    
    print(f"Prompt: {prompt}")
    output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=80,
        temperature=0.8,
        top_p=0.9
    )
    print("Generated:")
    print(output)