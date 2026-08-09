from transformers import Qwen2Config, Qwen2ForCausalLM
import torch
import random
import glob
from torch.utils.data import Dataset
import os
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B",trust_remote_code=True)
tokenizer.model_max_length = 10000000
if tokenizer.eos_token_id is not None:
    pad_token_id = tokenizer.eos_token_id
else:
    pad_token_id = 151643  
    tokenizer.eos_token = tokenizer.decode([pad_token_id]) 
tokenizer.pad_token = tokenizer.decode([pad_token_id])
tokenizer.pad_token_id = pad_token_id

model_config = Qwen2Config(
                    vocab_size=tokenizer.vocab_size,
                    hidden_size=2048,              
                    num_hidden_layers=24,           
                    num_attention_heads=16,       
                    num_key_value_heads=8,          
                    intermediate_size=int(2048 * 2.67), 
                    max_position_embeddings=2048, 
                    rms_norm_eps=1e-6,
                    hidden_dropout=0.1,
                    attention_dropout=0.1,
                )
model = Qwen2ForCausalLM(model_config).to(DEVICE)
model.config.use_cache = False          
#model.gradient_checkpointing_enable()   

training_args = TrainingArguments(
                    output_dir='./checkpoints',
                    logging_dir='./logs',
                    num_train_epochs=30,

                    learning_rate=1e-4,
                    weight_decay=0.01,
                    warmup_ratio=0.03,
                    lr_scheduler_type="cosine",
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=6,
                    gradient_accumulation_steps=4,
                    gradient_checkpointing=True,

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
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return {'input_ids': input_ids, 'labels': labels}

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    set_seed(42)
    
    #block_size = model.config.max_position_embeddings
    block_size = 2048

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
        torch.save(all_blocks,"checkpoints/data.pth")
    else:
        all_blocks=torch.load("checkpoints/data.pth",weights_only=False)
        print(f"loaded cache blocks: {len(all_blocks)}")

    random.shuffle(all_blocks)
    split = int(0.993 * len(all_blocks))
    train_blocks = all_blocks[:split]
    eval_blocks = all_blocks[split:]
    train_dataset = TextDataset(train_blocks)
    eval_dataset = TextDataset(eval_blocks)

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=collate_fn,
                )
    trainer.train()
    #trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-560/")
