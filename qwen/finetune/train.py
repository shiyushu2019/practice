# CUDA_VISIBLE_DEVICES=0,5 torchrun --nproc_per_node=2 --master_port=29501 train.py
# CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 train.py
# CUDA_VISIBLE_DEVICES=0,1,5 torchrun --nproc_per_node=3 --master_port=29599 train.py

import logging
from transformers import Qwen2Config, Qwen2ForCausalLM
import torch
from safetensors.torch import load_file
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
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
pad_token_id = tokenizer.pad_token_id

model_config = Qwen2Config(
                    vocab_size=len(tokenizer),
                    hidden_size=2048,   
                    intermediate_size=5504,
                    num_hidden_layers=20,           
                    num_attention_heads=16,       
                    num_key_value_heads=8,          
                    max_position_embeddings=768, 
                    attention_dropout=0.0,
                    rope_parameters={"rope_theta": 100000.0, "rope_type": "default"},
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False, 
                )
model_config._attn_implementation = "sdpa"  

training_args = TrainingArguments(
                    output_dir='./checkpoints',
                    max_steps=int(2e5),
                    warmup_steps=int(6e3),

                    learning_rate=3e-4,
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    per_device_train_batch_size=6,
                    gradient_accumulation_steps=15,
                    gradient_checkpointing=True,

                    save_total_limit=2,
                    logging_steps=10,
                    save_strategy="steps",
                    save_steps=500,
                    
                    fp16=True,
                    optim="adamw_8bit",
                    dataloader_num_workers=0,
                    max_grad_norm=1.0
                )

def collate_fn(batch):
    if len(batch) == 0:
        return {
            "input_ids": torch.tensor([[tokenizer.pad_token_id] * 768], dtype=torch.long),
            "labels": torch.tensor([[-100] * 768], dtype=torch.long),
            "attention_mask": torch.tensor([[0] * 768], dtype=torch.long),
        }
    input_ids = torch.stack([torch.tensor(x["input_ids"], dtype=torch.long) for x in batch])
    labels = torch.stack([torch.tensor(x["labels"], dtype=torch.long) for x in batch])
    attention_mask = torch.stack([torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch])
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

if __name__ == "__main__":

    #if torch.cuda.is_available():
    #    torch.backends.cuda.enable_flash_sdp(True)
    #    torch.backends.cuda.enable_mem_efficient_sdp(True)
    model = Qwen2ForCausalLM(model_config)
    missing, unexpected=model.load_state_dict(load_file('../pretrain/checkpoints/checkpoint-6000/model.safetensors'))
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    total_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"参数量: {total_b:.4f} B")
    print(f"Attention implementation: {model.config._attn_implementation}")

    from data import QADataset
    train_dataset=QADataset(path="/data1/dataset/jssnu005/sys/I_Wonder_Why-Chinese/general",
                            max_length=model.config.max_position_embeddings,
                            tokenizer=tokenizer)

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    data_collator=collate_fn,
                )
    # trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-200/")
    trainer.train()
    print("training finished")