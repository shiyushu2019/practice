# python inf.py --checkpoint "./checkpoints/checkpoint-3500/model.safetensors"

import threading
from transformers import AutoTokenizer, Qwen2ForCausalLM, TextIteratorStreamer
import argparse
import os
from safetensors.torch import load_file
from train import model_config

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
model = Qwen2ForCausalLM(model_config)
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
    流式生成文本，使用模型自带 KV cache，避免每步重算整段历史。
    """
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if max_ctx is not None and input_ids.shape[-1] >= max_ctx:
        raise ValueError(
            f"Prompt length ({input_ids.shape[-1]}) exceeds model max context ({max_ctx}). "
            "Please shorten the prompt or reduce max_new_tokens."
        )

    if max_ctx is not None:
        safe_new_tokens = min(max_new_tokens, max_ctx - input_ids.shape[-1])
        if safe_new_tokens <= 0:
            raise ValueError(
                f"No remaining context budget for generation: prompt length={input_ids.shape[-1]}, "
                f"max_context={max_ctx}."
            )
    else:
        safe_new_tokens = max_new_tokens

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=safe_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_k=top_k if (top_k is not None and temperature > 0) else None,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    try:
        for text in streamer:
            if text:
                print(text, end="", flush=True)
    finally:
        thread.join()

    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径（.pt 或 .bin）")
    parser.add_argument("--prompt", type=str, default="我，", help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.5, help="温度（0 为贪心）")
    parser.add_argument("--top_k", type=int, default=30, help="Top-k 采样（None 则关闭）")
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
    model=model.to(args.device)

    # 生成
    print(args.prompt, end="", flush=True)
    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    print()