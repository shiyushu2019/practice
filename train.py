import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from model import MyModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B",trust_remote_code=True)
model = MyModel(vocab_size=tokenizer.vocab_size,
                    num_layers=24,
                    d_model=2048,
                    num_heads=16,
                    num_kv_heads=8,
                    dropout=0.1,
                    max_seq_len=8192).to(DEVICE)

L = model.max_seq_len
causal_mask = torch.triu(torch.ones(L, L, dtype=torch.float32), diagonal=1)
causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)   # (1, 1, L, L)

training_args = TrainingArguments(
    output_dir='./checkpoints',
    logging_dir='./logs',
    num_train_epochs=1,

    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,

    eval_strategy="steps",
    eval_steps=1000,
    metric_for_best_model="eval_loss",
    greater_is_better=False, 
    load_best_model_at_end=True, 
    save_total_limit=3,
    logging_steps=10,
    save_strategy="best",
    
    fp16=False,
    dataloader_num_workers=4,
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
    # 复制因果掩码到 batch 维度
    attn_mask = causal_mask.expand(B, -1, -1, -1).contiguous()
    # 无 padding，全 False
    key_padding_mask = torch.zeros(B, L, dtype=torch.bool)
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attn_mask': attn_mask,
        'key_padding_mask': key_padding_mask
    }

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        logits = model(
            input_ids=inputs['input_ids'],
            attn_mask=inputs['attn_mask'],
            key_padding_mask=inputs['key_padding_mask']
        )
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            inputs['labels'].view(-1),
            ignore_index=-100
        )
        return (loss, logits) if return_outputs else loss

if __name__ == "__main__":
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    block_size = model.max_seq_len         
    blocks = [tokens[i:i+block_size] for i in range(0, len(tokens), block_size)
              if len(tokens[i:i+block_size]) == block_size]
    random.shuffle(blocks)
    split = int(0.9 * len(blocks))
    train_blocks = blocks[:split]
    eval_blocks = blocks[split:]

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
