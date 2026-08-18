# CUDA_VISIBLE_DEVICES=0,1,5 torchrun --nproc_per_node=3 --master_port=29588 train.py
# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 torchrun --nproc_per_node=7 --master_port=29588 train.py

from transformers import AutoModelForCausalLM
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)

path="/data1/dataset/jssnu005/sys/qwen3-14B"
tokenizer = AutoTokenizer.from_pretrained(path)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,    # 因果语言建模
    inference_mode=False,     # 训练模式必须为 False
    r=64, 
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj", 
        "o_proj",          
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ],
    lora_dropout=0.05,               
    bias="none",  # 不训练偏置项，节省显存
)

training_args = TrainingArguments(
                    output_dir='./checkpoints',
                    warmup_steps=1360,
                    num_train_epochs=1,

                    learning_rate=5e-5,
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=3,
                    gradient_checkpointing=True,

                    save_total_limit=3,
                    logging_steps=10,
                    save_strategy="steps",
                    save_steps=1000,
                    
                    #fp16=True,
                    bf16=True,
                    optim="adamw_8bit",
                    dataloader_num_workers=0,
                    max_grad_norm=1.0,
                    ddp_find_unused_parameters=False
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
    model = AutoModelForCausalLM.from_pretrained(path,
                                                dtype= torch.bfloat16,
                                                attn_implementation= "sdpa",
                                                use_cache=False, # 与 gradient_checkpointing 
                                            )

    #if torch.cuda.is_available():
    #    torch.backends.cuda.enable_flash_sdp(True)
    #    torch.backends.cuda.enable_mem_efficient_sdp(True)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from data import QADataset # 指令遵循数据集
    train_dataset=QADataset(path="/data1/dataset/jssnu005/sys/I_Wonder_Why-Chinese/general",
                            max_length=768,
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