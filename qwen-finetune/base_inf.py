
# python base_inf.py  --prompt "风从玉兰树梢滑下来，翻开了膝上那本旧笔记。姑姑的字迹在某一页停下：“今天铜铃响了三 次，每一次我都以为是春天。”我抬头，檐角的铜铃正轻轻转着，像在回应什么。阁楼的门半掩着，一束光斜斜地铺进来，刚好照亮了那一页空白的纸。"
# export CUDA_VISIBLE_DEVICES=2

import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import argparse

path="/data1/dataset/jssnu005/sys/qwen2.5"
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=1.0,
    top_p=None,
    device="cuda"
):
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

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=safe_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p if (top_p is not None and temperature > 0) else None,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.15,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    try:
        for text in streamer:
            if text:
                print(text, end="", flush=True)
    finally:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="我，", help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度（0 为贪心）")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p 采样（None 则关闭）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    model=model.to(args.device)

    # 生成
    print(args.prompt, end="", flush=True)
    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device
    )
    print()