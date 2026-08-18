# export CUDA_VISIBLE_DEVICES=2

import argparse
import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

base_path = "/data1/dataset/jssnu005/sys/qwen3-0.6B" # [SMOKE]
tokenizer = AutoTokenizer.from_pretrained(base_path)

def get_latest_checkpoint(base_dir="./checkpoints"):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {base_dir}")
    max_num = -1
    max_path = None
    for item in os.listdir(base_dir):
        if item.startswith("checkpoint-"):
            try:
                num = int(item.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if num > max_num:
                max_num = num
                max_path = os.path.join(base_dir, item)
    if max_path is None:
        raise FileNotFoundError(f"No valid checkpoint found in: {base_dir}")
    return max_path


def prepare_input_ids(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    elif isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    elif isinstance(encoded, torch.Tensor):
        input_ids = encoded
    else:
        raise TypeError(
            f"Unexpected chat template output type: {type(encoded)}. "
            "Expected BatchEncoding/dict-like object with an input_ids field."
        )

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids.to(device)


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=1.0,
    top_p=None,
    device="cuda",
):
    model.eval()
    input_ids = prepare_input_ids(tokenizer, prompt, device)

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

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        stop_strings=['<|endoftext|>'],
        tokenizer=tokenizer,
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=safe_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p if (top_p is not None and temperature > 0) else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.15,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
    thread.start()

    try:
        for text in streamer:
            if text:
                print(text, end="", flush=True)
    finally:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.8, help="温度（0 为贪心）")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 采样（None 则关闭）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    torch_dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch_dtype,
        attn_implementation="sdpa",
        use_cache=True,
    )

    model = model.to(args.device)

    while True:
        prompt = input("user:")
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=768,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        print()