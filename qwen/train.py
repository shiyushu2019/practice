from transformers import Qwen2Config, Qwen2ForCausalLM
import torch
import random
import glob
from torch.utils.data import Dataset
import os
from datasets import load_dataset
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
                    intermediate_size=11008,
                    max_position_embeddings=2048, 
                    rms_norm_eps=1e-6,
                    attention_dropout=0.0,
                    rope_parameters={"rope_theta": 100000.0, "rope_type": "default"},
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,  # 训练时关闭缓存
                )
model = Qwen2ForCausalLM.from_config(model_config).to(DEVICE)    

training_args = TrainingArguments(
                    output_dir='./checkpoints',
                    logging_dir='./logs',
                    #num_train_epochs=30,
                    max_steps=100000, # steps to train
                    warmup_steps=3000,

                    learning_rate=1e-4,
                    weight_decay=0.01,
                    #warmup_ratio=0.03,
                    lr_scheduler_type="cosine",
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=6,
                    gradient_accumulation_steps=4,
                    gradient_checkpointing=True,

                    eval_strategy="steps",
                    eval_steps=5000,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False, 
                    load_best_model_at_end=True, 
                    save_total_limit=3,
                    logging_steps=30,
                    save_strategy="steps",
                    save_steps=5000,
                    
                    fp16=True,
                    optim="adamw_8bit",
                    dataloader_num_workers=0,
                    max_grad_norm=1.0
                )

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = input_ids.clone()
    labels[labels == pad_token_id] = -100
    return {'input_ids': input_ids, 'labels': labels}

if __name__ == "__main__":

    data_dir = "/data1/dataset/jssnu005/sys/MNBVC"
    file_list = glob.glob(f"{data_dir}/**/*.jsonl.gz", recursive=True)
    random.shuffle(file_list) 
    dataset = load_dataset(
                        "json",
                        data_files=file_list,
                        split="train",
                        streaming=True,
                    )
    # 定义允许的纯文本/代码扩展名（白名单）
    ALLOWED_EXTS = {
        '.py', '.java', '.c', '.cpp', '.h', '.js', '.ts', '.go', '.rs',
        '.md', '.txt', '.json', '.xml', '.html', '.css', '.sh', '.r',
        '.ipynb', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.csv',
        '.xml', '.xsl', '.xsd', '.dtd', '.svg', '.rss', '.atom',
        '.pl', '.pm', '.t', '.pod', '.cgi', '.rb', '.erb', '.rake',
        '.php', '.phpt', '.phtml', '.lua', '.m', '.mm', '.swift',
        '.vb', '.vbs', '.ps1', '.bat', '.cmd', '.bash', '.zsh',
        '.fish', '.sql', '.db', '.sqlite', '.lisp', '.cl', '.el',
        '.scala', '.kt', '.kts', '.dart', '.jl', '.r', '.rmd',
    }
    def keep_allowed_ext(example):
        """返回 True 表示保留，False 表示过滤掉"""
        ext = example.get('ext', '')
        # 转为小写并去除首尾空格，确保匹配
        ext = ext.strip().lower()
        # 如果扩展名不在白名单中，则丢弃
        return ext in ALLOWED_EXTS
    
    dataset = dataset.filter(keep_allowed_ext)

    def tokenize_and_chunk(example):
        block_size = model.config.max_position_embeddings
        stride = block_size
        text = example.get('text', '')
        if not text:
            return {"input_ids": []}  # 空文本返回空列表，后续过滤掉
        tokens = tokenizer.encode(text, add_special_tokens=False)
        blocks = []
        if len(tokens) <= block_size:
            # 短文本：补齐到 block_size
            block = tokens + [pad_token_id] * (block_size - len(tokens))
            blocks.append(block)
        else:
            # 长文本：滑动窗口切块
            for start in range(0, len(tokens) - block_size + 1, stride):
                blocks.append(tokens[start:start + block_size])
            # 最后不足步长时，取最后 block_size 个 token（确保覆盖尾部）
            if (len(tokens) - block_size) % stride != 0:
                blocks.append(tokens[-block_size:])
        return {"input_ids": blocks}  # 返回列表，HF 会自动展开为多个样本

    dataset = dataset.map(
        tokenize_and_chunk,
        batched=False,
        remove_columns=dataset.column_names,
        # remove_columns=['来源', '仓库名', 'path', '文件名', 'ext', 'size', '原始编码', 'md5', 'text', '时间'], # keys are known
    )
    dataset = dataset.filter(lambda x: len(x['input_ids']) > 0)
    
    eval_dataset = dataset.take(10000)
    train_dataset = dataset.skip(10000)

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=collate_fn,
                )
    trainer.train()
    #trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-560/")