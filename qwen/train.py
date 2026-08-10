# export CUDA_VISIBLE_DEVICES=2

from transformers import Qwen2Config, Qwen2ForCausalLM
import torch
from torch.utils.data import IterableDataset
import random
import glob
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

counters = {
    'bad_ext_examples': 0,
    'tokenize_error_examples': 0,
    'skipped_input_ids_examples': 0,
    'filtered_empty_input_ids': 0,
}

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
                    intermediate_size=14080,
                    num_hidden_layers=30,           
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
    max_len = max(len(seq) for seq in batch)
    if max_len == 0:
        raise ValueError("All input_ids in batch are empty sequences (length 0). Check tokenizer output and filtering.")
    padded_input_ids = []
    for seq in batch:
        pad_len = max_len - len(seq)
        padded_seq = seq + [tokenizer.pad_token_id] * pad_len
        padded_input_ids.append(padded_seq)
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': input_ids, 'labels': labels}

class BlocksDataset(IterableDataset):
    def __init__(self,
                data_dir = "/data1/dataset/jssnu005/sys/MNBVC"
                ):
        self.buffer=[]
        self.pointer=self._get_pointer(data_dir = data_dir,)

    def __iter__(self):
        if len(self.buffer)>0:
            yield self.buffer.pop()
        else:
            while not len(self.buffer)>0:
                try:
                    get = next(self.pointer)['input_ids']
                except StopIteration:
                    raise StopIteration
                if isinstance(get, list) and len(get) > 0:
                    if all(isinstance(tok, int) for tok in get):   # [1,2,3....]
                        self.buffer.append(get)
                    elif all(isinstance(chunk, list) and all(isinstance(tok, int) for tok in chunk) for chunk in get): # [[1,2,3....], [1,2,3,...]]
                        # seq is a list of chunks; filter out any empty chunks (eg. [[]])
                        filtered_chunks = [chunk for chunk in get if isinstance(chunk, list) and len(chunk) > 0 and all(isinstance(tok, int) for tok in chunk)]
                        if len(filtered_chunks) > 0:
                            self.buffer.extend(filtered_chunks)
                        else:
                            counters['skipped_input_ids_examples'] += 1
                    else:
                        counters['skipped_input_ids_examples'] += 1
                else:
                    print(f"[error] collate_fn skipped invalid input_ids: {get}")
                    counters['skipped_input_ids_examples'] += 1
            yield self.buffer.pop()

    def _get_pointer(self,
                    data_dir = "/data1/dataset/jssnu005/sys/MNBVC",
                ):
        
        file_list = glob.glob(f"{data_dir}/**/*.jsonl.gz", recursive=True)
        print(f"[BlocksDataset] Number of files: {len(file_list)}")
    
        random.shuffle(file_list) 
        dataset = load_dataset(
                            "json",
                            data_files=file_list,
                            split="train",
                            streaming=True,
                        )
    
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
            try:
                ext = example.get('ext', '')
                if not isinstance(ext, str):
                    ext = str(ext) if ext is not None else ''
                ext = ext.strip().lower()
                if not ext.startswith('.') and ext != '':
                    ext = '.' + ext
                if ext not in ALLOWED_EXTS:
                    counters['bad_ext_examples'] += 1
                return ext in ALLOWED_EXTS
            except Exception as e:
                counters['bad_ext_examples'] += 1
                print(f"[error] keep_allowed_ext failed: {e}; example={example}")
                return False
        
        dataset = dataset.filter(keep_allowed_ext)
    
        def tokenize_and_chunk(examples):
            try:
                block_size = model.config.max_position_embeddings
                texts = examples.get('text', [])
                if texts is None:
                    texts = []
                if not isinstance(texts, list):
                    texts = [texts]
    
                chunked_input_ids = []
                for text in texts:
                    if text is None:
                        continue
                    if not isinstance(text, str):
                        text = str(text)
                    if len(text) == 0:
                        counters['filtered_empty_input_ids'] += 1
                        continue
    
                    encodings = tokenizer(
                        text,
                        return_overflowing_tokens=True,
                        max_length=block_size,
                        stride=0,
                        truncation=True,
                        add_special_tokens=False,
                    )
                    chunk_ids = encodings.get('input_ids', [])
                    if isinstance(chunk_ids, list):
                        if len(chunk_ids) == 0:
                            continue
                        # chunk_ids can be either a flat list of ints (single chunk) or a list of chunks
                        if all(isinstance(tok, int) for tok in chunk_ids):
                            if len(chunk_ids) > 0:
                                chunked_input_ids.append(chunk_ids)
                        else:
                            for chunk in chunk_ids:
                                if isinstance(chunk, list) and len(chunk) > 0 and all(isinstance(tok, int) for tok in chunk):
                                    chunked_input_ids.append(chunk)
                return {'input_ids': chunked_input_ids}
            except Exception as e:
                counters['tokenize_error_examples'] += 1
                identifier = examples.get('path') or examples.get('文件名') or examples.get('md5') or '<unknown>'
                print(f"[error] tokenize_and_chunk failed for {identifier}: {e}")
                return {'input_ids': []}
    
        dataset = dataset.map(
            tokenize_and_chunk,
            batched=True,
            batch_size=100,
            remove_columns=dataset.column_names,
            #remove_columns=['来源', '仓库名', 'path', '文件名', 'ext', 'size', '原始编码', 'md5', 'text', '时间'], # keys are known
        )
        dataset = dataset.filter(
            lambda x: isinstance(x.get('input_ids', None), list) and len(x['input_ids']) > 0
        )

        return iter(dataset) # return the pointer


if __name__ == "__main__":

    total_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"参数量: {total_b:.4f} B")

    train_dataset=BlocksDataset()

    trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    #eval_dataset=eval_dataset,
                    data_collator=collate_fn,
                )
    trainer.train()

    print("[stats] bad_ext_examples=", counters['bad_ext_examples'])
    print("[stats] tokenize_error_examples=", counters['tokenize_error_examples'])
    print("[stats] filtered_empty_input_ids=", counters['filtered_empty_input_ids'])
    print("[stats] skipped_input_ids_examples=", counters['skipped_input_ids_examples'])
    #trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-560/")