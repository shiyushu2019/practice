# export CUDA_VISIBLE_DEVICES=2

import logging
from transformers import Qwen2Config, Qwen2ForCausalLM
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

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
                    hidden_size=2560,   
                    intermediate_size=12288,
                    num_hidden_layers=24,           
                    num_attention_heads=20,       
                    num_key_value_heads=10,          
                    max_position_embeddings=4096, 
                    #rms_norm_eps=1e-6,
                    attention_dropout=0.0,
                    rope_parameters={"rope_theta": 100000.0, "rope_type": "default"},
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False,  # 训练时关闭缓存
                )
model = Qwen2ForCausalLM(model_config).to(DEVICE)    

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
                    per_device_train_batch_size=1,
                    #per_device_eval_batch_size=6,
                    gradient_accumulation_steps=8,
                    gradient_checkpointing=True,

                    #eval_strategy="steps",
                    #eval_steps=5000,
                    #metric_for_best_model="eval_loss",
                    #greater_is_better=False, 
                    #load_best_model_at_end=True, 
                    save_total_limit=3,
                    logging_steps=5,
                    save_strategy="steps",
                    save_steps=5000,
                    
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

    total_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"参数量: {total_b:.4f} B")

    from data import BlocksDataset
    train_dataset=BlocksDataset(data_dir = "/data1/dataset/jssnu005/sys/MNBVC",
                                logger=logger,
                                model=model,
                                tokenizer=tokenizer)

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    #eval_dataset=eval_dataset,
                    data_collator=collate_fn,
                )
    trainer.train()

    logger.info("training finished")
    #trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-560/")