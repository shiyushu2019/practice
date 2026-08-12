from torch.utils.data import IterableDataset
import random
import glob
import torch.distributed as dist
import json

class BlocksDataset(IterableDataset):
    def __init__(self,
                data_dir = "/data1/dataset/jssnu005/sys/skypile/data",
                *,logger,model_max_len,tokenizer,
                ):
        self.logger=logger
        self.model_max_len=model_max_len
        self.tokenizer=tokenizer
        self.data_dir=data_dir

    def __iter__(self):
        block_size = self.model_max_len
        pad_id=self.tokenizer.pad_token_id
        try:
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
            file_list = sorted(glob.glob(f"{self.data_dir}/*.jsonl", recursive=True))
            if len(file_list) == 0:
                print(f"No files found in {self.data_dir}; dataset is empty.")
                raise FileNotFoundError
            print(f"[BlocksDataset] Number of files: {len(file_list)}")
            random.seed(123) # 确保多个进程随机结果一样
            random.shuffle(file_list)
            file_list = [f for i, f in enumerate(file_list) if i % world_size == rank]
        except Exception as e:
            print(f"failed to load .jsonl files: {e}") 
            exit(1) # 前期数据预处理出错，直接退出

        for fpath in file_list:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            content=json.loads(line)
                            try:
                                text=content['text']
                            except Exception as e:
                                self.logger.error(f"[__iter__] getting a 'text' key failed: {e}; content={content}")
                                continue # 保证数据流不崩溃
                            try:
                                encodings = self.tokenizer(
                                                            text,
                                                            return_overflowing_tokens=True,
                                                            max_length=block_size,
                                                            stride=0,
                                                            truncation=True,
                                                            add_special_tokens=False,
                                                            padding='max_length',
                                                        )
                            except Exception as e:
                                self.logger.error(f"[__iter__] encoding failed: {e}; text={text}")
                                continue # 保证数据流不崩溃
                            chunk_ids = encodings['input_ids']
                            if len(chunk_ids)>0 and all(isinstance(tok, int) for tok in chunk_ids): # 只返回一个token：[1,2,3,....]
                                chunk=chunk_ids
                                if isinstance(chunk, list) and all(tok==pad_id for tok in chunk):
                                    self.logger.error(f"[__iter__] do not know why the trunk is all pad_id: {chunk}")
                                    continue # 保证数据流不崩溃
                                yield chunk
                            else:  # 返回多个token：[[1,2,3,....], ....]
                                for chunk in chunk_ids:
                                    try:
                                        if isinstance(chunk, list) and all(tok==pad_id for tok in chunk):
                                            self.logger.error(f"[__iter__] do not know why the trunk is all pad_id: {chunk}")
                                            continue # 保证数据流不崩溃
                                        elif isinstance(chunk, list) and len(chunk) > 0 and all(isinstance(tok, int) for tok in chunk):
                                            yield chunk
                                        else:
                                            self.logger.error(f"[__iter__] invalid tokenized chunk format: {chunk}")
                                            continue # 保证数据流不崩溃
                                    except Exception as e:
                                        self.logger.error(f"[__iter__] one chunk failed: {e}; chunk={chunk}")
                                        continue # 保证数据流不崩溃
                        except Exception as e:
                            self.logger.error(f"[__iter__] one json line failed: {e}; line={line}")
                            continue # 保证数据流不崩溃
            except Exception as e:
                self.logger.error(f"[__iter__] one .jsonl file failed: {e}; file={fpath}")
                continue # 保证数据流不崩溃

if __name__ == "__main__":
    from train import logger,  tokenizer

    dataset = BlocksDataset(logger=logger, model_max_len=4096, tokenizer=tokenizer)
    it = iter(dataset)

    
    cnt=0
    Min_len=4096
    avg_len=0
    for x in dataset:
        cnt+=1
        Min_len=min(Min_len,len(x))
        avg_len+=len(x)
    avg_len/=cnt
    print(cnt,Min_len,avg_len)
    """
    for i in range(1000):
        try:
            block = next(it)
        except StopIteration:
            print("数据流已结束")
            break

        # 解码为文本
        text = tokenizer.decode(block, skip_special_tokens=True)
        print(f"\n========== Block {i} (tokens: {len(block)}) ==========")
        print(text)
        input("按 Enter 键继续下一个块...")
    """