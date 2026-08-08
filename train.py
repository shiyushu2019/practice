# export CUDA_VISIBLE_DEVICES=2
# nohup python train.py 2>&1 &

import torch
import torch.nn as nn
import random
import glob
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from model import MyModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B",trust_remote_code=True)
tokenizer.model_max_length = 10000000
if tokenizer.eos_token_id is not None:
    pad_token_id = tokenizer.eos_token_id
else:
    pad_token_id = 151643  
    tokenizer.eos_token = tokenizer.decode([pad_token_id]) 
tokenizer.pad_token = tokenizer.decode([pad_token_id])
tokenizer.pad_token_id = pad_token_id

model = MyModel(vocab_size=tokenizer.vocab_size,
                    num_layers=24,
                    d_model=2048,
                    num_heads=16,
                    num_kv_heads=8,
                    dropout=0.1,
                    max_seq_len=2048).to(DEVICE)

training_args = TrainingArguments(
                    output_dir='./checkpoints',
                    logging_dir='./logs',
                    num_train_epochs=30,

                    learning_rate=1e-4,
                    weight_decay=0.01,
                    warmup_ratio=0.03,
                    lr_scheduler_type="cosine",
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=5,

                    #eval_strategy="epoch",
                    eval_strategy="steps",
                    eval_steps=500,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False, 
                    load_best_model_at_end=True, 
                    save_total_limit=1,
                    logging_steps=30,
                    save_strategy="best",
                    
                    fp16=True,
                    optim="adamw_8bit",
                    dataloader_num_workers=2,
                    max_grad_norm=1.0
                    )

class TextDataset(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks
    def __len__(self):
        return len(self.blocks)
    def __getitem__(self, idx):
        return {'input_ids': torch.tensor(self.blocks[idx], dtype=torch.long)}

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])  # (B, L)
    B, L = input_ids.shape
    # 生成 labels：向右移位，最后一位用 -100 忽略
    labels = torch.cat([input_ids[:, 1:],
                        torch.full((B, 1), -100, dtype=torch.long)], dim=1)
    labels[labels == pad_token_id] = -100
    return {'input_ids': input_ids, 'labels': labels}

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = model(
            input_ids=inputs['input_ids']
        )
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            inputs['labels'].view(-1),
            ignore_index=-100
        )
        return (loss, logits) if return_outputs else loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 提取模型需要的参数
        input_ids = inputs['input_ids']
        labels = inputs.get('labels')

        with torch.no_grad():
            # 前向传播，获取 logits（不计算 loss，但我们会手动计算）
            logits = model(
                input_ids=input_ids,
            )
            # 计算损失（如果提供了 labels）
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

        # 根据 prediction_loss_only 决定返回什么
        if prediction_loss_only:
            return (loss, None, None)
        else:
            return (loss, logits, labels)

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    
    block_size = model.max_seq_len

    if not os.path.exists("checkpoints/data.pth"):
        print(f"loading books")
        all_blocks = []

        book_cnt=0
        for filepath in sorted(glob.glob("./data/*.txt")):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = tokenizer.encode(text, add_special_tokens=False)

            for i in range(0, len(tokens), block_size):
                block = tokens[i:i+block_size]
                if len(block) < block_size:
                    # 补全到 block_size
                    block = block + [pad_token_id] * (block_size - len(block))
                all_blocks.append(block)
            
            book_cnt+=1
        print(f"having {book_cnt} books")
        torch.save(all_blocks,"data/data.pth")
    else:
        all_blocks=torch.load("checkpoints/data.pth",weights_only=False)
        print(f"loaded cache blocks: {len(all_blocks)}")

    random.shuffle(all_blocks)
    
    split = int(0.95 * len(all_blocks))
    train_blocks = all_blocks[:split]
    eval_blocks = all_blocks[split:]

    train_dataset = TextDataset(train_blocks)
    eval_dataset = TextDataset(eval_blocks)

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    #trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-560/")
