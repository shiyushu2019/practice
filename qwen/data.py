
from torch.utils.data import IterableDataset
import random
import glob
from datasets import load_dataset
import gzip,json
import os

class BlocksDataset(IterableDataset):
    def __init__(self,
                data_dir = "/data1/dataset/jssnu005/sys/MNBVC",
                sub_name=[], # 排除一级文件夹
                *,logger,model_max_len,tokenizer,
                ):
        self.logger=logger
        self.model_max_len=model_max_len
        self.tokenizer=tokenizer
        self.data_dir=data_dir
        self.sub_name=sub_name

    def _check_ext(self,ext):
        # 返回 True 表示保留，False 表示过滤掉
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
        try:
            if ext==None:
                self.logger.warning(f"[_check_ext] There is a line with out ext column")
                return True # 没有扩展名时，信任为文本文件
            if not isinstance(ext, str):
                ext = str(ext) if ext is not None else ''
            ext = ext.strip().lower()
            if not ext.startswith('.') and ext != '':
                ext = '.' + ext
            if ext not in ALLOWED_EXTS:
                pass # 预计有很多，不记录
                # self.logger.info(f"filtered bad extension: {ext} from example {example.get('path', '<no path>')}")
            return ext in ALLOWED_EXTS
        except Exception as e:
            self.logger.error(f"[_check_ext] _check_ext function failed: {e}; example={ext}")
            return False

    def __iter__(self):
        block_size = self.model_max_len
        try:
            file_list = glob.glob(f"{self.data_dir}/**/*.jsonl.gz", recursive=True)
            if len(file_list) == 0:
                print(f"No jsonl.gz files found in {self.data_dir}; dataset is empty.")
                raise FileNotFoundError
            filtered_list = []
            for f in file_list:
                rel = os.path.relpath(f, self.data_dir)
                parts = rel.split(os.sep)
                if parts and parts[0] not in self.sub_name:
                    continue  # 跳过该文件
                filtered_list.append(f)
            file_list = filtered_list
            print(f"[BlocksDataset] Number of files: {len(file_list)}")
            if len(file_list) == 0:
                print(f"After fiter, no jsonl.gz files found in {self.data_dir}; dataset is empty.")
                raise FileNotFoundError
            random.shuffle(file_list)
        except Exception as e:
            print(f"failed to load .gz files: {e}") 
            exit(1) # 前期数据预处理出错，直接退出

        for fpath in file_list:
            try:
                with gzip.open(fpath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            raw = json.loads(line)
                            text= raw.get('text', '')
                            ext= raw.get('ext', None)
                            if not self._check_ext(ext):
                                continue # 预计较多，不记录日志
                            encodings = self.tokenizer(
                                                        text,
                                                        return_overflowing_tokens=True,
                                                        max_length=block_size,
                                                        stride=0,
                                                        truncation=True,
                                                        add_special_tokens=False,
                                                    )
                            chunk_ids = encodings['input_ids']
                            if len(chunk_ids)>0 and all(isinstance(tok, int) for tok in chunk_ids): # 只返回一个token：[1,2,3,....]
                                yield chunk_ids
                            else:  # 返回多个token：[[1,2,3,....], ....]
                                for chunk in chunk_ids:
                                    try:
                                        if isinstance(chunk, list) and len(chunk) > 0 and all(isinstance(tok, int) for tok in chunk):
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
                self.logger.error(f"[__iter__] one .gz file failed: {e}; file={fpath}")
                continue # 保证数据流不崩溃

if __name__ == "__main__":
    from train import logger,  tokenizer

    dataset = BlocksDataset(logger=logger, model_max_len=100, tokenizer=tokenizer, sub_name=['book'])
    it = iter(dataset)

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