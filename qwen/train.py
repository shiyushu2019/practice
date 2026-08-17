# export CUDA_VISIBLE_DEVICES=2
# CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 --master_port=29501 train.py
# CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 train.py
# CUDA_VISIBLE_DEVICES=0,1,5 torchrun --nproc_per_node=3 --master_port=29599 train.py

import logging
from transformers import Qwen2Config, Qwen2ForCausalLM
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)

logger = logging.getLogger("data_processing")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler("data.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
tokenizer.model_max_length = 10000000
if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
    # add a dedicated pad token (will update tokenizer.vocab_size)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
pad_token_id = tokenizer.pad_token_id

model_config = Qwen2Config(
                    #vocab_size=tokenizer.vocab_size,
                    vocab_size=len(tokenizer),
                    hidden_size=2048,   
                    intermediate_size=5504,
                    num_hidden_layers=20,           
                    num_attention_heads=16,       
                    num_key_value_heads=8,          
                    max_position_embeddings=768, 
                    #rms_norm_eps=1e-6,
                    attention_dropout=0.0,
                    rope_parameters={"rope_theta": 100000.0, "rope_type": "default"},
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,  # 训练时关闭缓存
                )
# model_config._attn_implementation = "sdpa"  

training_args = TrainingArguments(
                    output_dir='./checkpoints',
                    #num_train_epochs=30,
                    max_steps=int(2e5),
                    warmup_steps=int(6e3),

                    learning_rate=5e-5,
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    per_device_train_batch_size=6,
                    gradient_accumulation_steps=15,
                    #gradient_accumulation_steps=1,
                    gradient_checkpointing=True,

                    save_total_limit=5,
                    logging_steps=10,
                    save_strategy="steps",
                    save_steps=500,
                    
                    fp16=True,
                    optim="adamw_8bit",
                    dataloader_num_workers=0,
                    max_grad_norm=1.0
                )

def collate_fn(batch):
    valid_batch = []
    for seq in batch:
        if isinstance(seq, list) and len(seq) > 0 and all(isinstance(tok, int) for tok in seq):
            valid_batch.append(seq)
        else:
            logger.warning(f"skipped invalid sequence in collate_fn: {seq}")
    if len(valid_batch) == 0:
        logger.warning("collate_fn received no valid sequences, using one pad-only fallback example.")
        valid_batch = [[tokenizer.pad_token_id]]
    max_len = max(len(seq) for seq in valid_batch)
    padded_input_ids = []
    for seq in valid_batch:
        pad_len = max_len - len(seq)
        padded_seq = seq + [tokenizer.pad_token_id] * pad_len
        padded_input_ids.append(padded_seq)
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': input_ids, 'labels': labels}

if __name__ == "__main__":

    #if torch.cuda.is_available():
    #    torch.backends.cuda.enable_flash_sdp(True)
    #    torch.backends.cuda.enable_mem_efficient_sdp(True)
    model = Qwen2ForCausalLM(model_config)

    from safetensors.torch import load_file
    state_dict = load_file("/data1/code/jssnu005/sys/practice/qwen/checkpoints/checkpoint-13500/model.safetensors")
    miss, _ = model.load_state_dict(state_dict, strict=False)
    print("miss:", miss)
    # 26-8-17 7%开始，放弃调度器和优化器

    total_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"参数量: {total_b:.4f} B")
    #print(f"Attention implementation: {model.config._attn_implementation}")

    from jsonl_data import BlocksDataset
    train_dataset=BlocksDataset(data_dir = "/data1/dataset/jssnu005/sys/skypile/data",
                                logger=logger,
                                model_max_len=model.config.max_position_embeddings,
                                tokenizer=tokenizer)

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    #eval_dataset=eval_dataset,
                    data_collator=collate_fn,
                )
    # trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-200/")
    trainer.train()
    print("training finished")