
from torch.utils.data import IterableDataset
import random
import glob
from datasets import load_dataset

class BlocksDataset(IterableDataset):
    def __init__(self,
                data_dir = "/data1/dataset/jssnu005/sys/MNBVC",
                *,logger,model,tokenizer
                ):
        self.buffer=[]
        self.logger=logger
        self.model=model
        self.tokenizer=tokenizer
        self.pointer=self._get_pointer(data_dir = data_dir,)

    def __iter__(self):
        while True:
            try:
                while not len(self.buffer) > 0:
                    try:
                        item = next(self.pointer)
                    except StopIteration:
                        return
                    if not isinstance(item, dict):
                        self.logger.warning(f"skipped non-dict item from pointer: {type(item)}")
                        continue
                    get = item.get('input_ids')
                    sequences = self._normalize_input_ids(get)
                    if len(sequences) > 0:
                        self.buffer.extend(sequences)
                    else:
                        self.logger.warning(f"skipped invalid input_ids structure: {type(get)} {get}")
                yield self.buffer.pop()       
            except Exception as e:
                self.logger.error(f"failed reading example from pointer: {e}")
                continue

    def _normalize_input_ids(self, input_ids):
        if isinstance(input_ids, tuple):
            input_ids = list(input_ids)
        if not isinstance(input_ids, list):
            self.logger.warning(f"invalid input_ids type: {type(input_ids)}")
            return []
        if len(input_ids) == 0:
            self.logger.warning("skipped empty input_ids list")
            return []
        if all(isinstance(tok, int) for tok in input_ids):
            return [input_ids]
        sequences = []
        for chunk in input_ids:
            if isinstance(chunk, tuple):
                chunk = list(chunk)
            if isinstance(chunk, list) and len(chunk) > 0 and all(isinstance(tok, int) for tok in chunk):
                sequences.append(chunk)
            else:
                self.logger.warning(f"invalid chunk in input_ids list: {chunk}")
        return sequences

    def _get_pointer(self,
                    data_dir = "/data1/dataset/jssnu005/sys/MNBVC",
                ):
        
        file_list = glob.glob(f"{data_dir}/**/*.jsonl.gz", recursive=True)
        print(f"[BlocksDataset] Number of files: {len(file_list)}")
        if len(file_list) == 0:
            self.logger.warning(f"No jsonl.gz files found in {data_dir}; dataset is empty.")
            return iter([])

        random.shuffle(file_list)
        try:
            dataset = load_dataset(
                                "json",
                                data_files=file_list,
                                split="train",
                                streaming=True,
                            )
        except Exception as e:
            self.logger.error(f"failed to load dataset from files: {e}")
            return iter([])
    
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
                if not isinstance(example, dict):
                    self.logger.warning(f"skipped non-dict example during extension filtering: {type(example)}")
                    return False
                ext = example.get('ext', '')
                if not isinstance(ext, str):
                    ext = str(ext) if ext is not None else ''
                ext = ext.strip().lower()
                if not ext.startswith('.') and ext != '':
                    ext = '.' + ext
                if ext not in ALLOWED_EXTS:
                    pass
                    # self.logger.info(f"filtered bad extension: {ext} from example {example.get('path', '<no path>')}")
                return ext in ALLOWED_EXTS
            except Exception as e:
                self.logger.error(f"keep_allowed_ext failed: {e}; example={example}")
                return False
        
        dataset = dataset.filter(keep_allowed_ext)
    
        def tokenize_and_chunk(examples):
            try:
                block_size = self.model.config.max_position_embeddings
                texts = examples.get('text', []) if isinstance(examples, dict) else []
                if texts is None:
                    texts = []
                if not isinstance(texts, list):
                    texts = [texts]

                chunked_input_ids = []
                for text in texts:
                    if text is None:
                        continue
                    if isinstance(text, list):
                        self.logger.warning(f"received unexpected list in text field: {text}")
                        continue
                    if not isinstance(text, str):
                        try:
                            text = str(text)
                        except Exception as e:
                            self.logger.warning(f"failed to stringify text field: {e}; value={text}")
                            continue
                    if len(text) == 0:
                        self.logger.info("filtered empty text field")
                        continue

                    encodings = self.tokenizer(
                        text,
                        return_overflowing_tokens=True,
                        max_length=block_size,
                        stride=0,
                        truncation=True,
                        add_special_tokens=False,
                    )
                    chunk_ids = encodings.get('input_ids', [])
                    if isinstance(chunk_ids, tuple):
                        chunk_ids = list(chunk_ids)
                    if isinstance(chunk_ids, list):
                        if len(chunk_ids) == 0:
                            self.logger.debug("filtered empty tokenized input_ids chunk")
                            continue
                        if all(isinstance(tok, int) for tok in chunk_ids):
                            chunked_input_ids.append(chunk_ids)
                        else:
                            for chunk in chunk_ids:
                                if isinstance(chunk, list) and len(chunk) > 0 and all(isinstance(tok, int) for tok in chunk):
                                    chunked_input_ids.append(chunk)
                                else:
                                    self.logger.warning(f"invalid tokenized chunk format: {chunk}")
                    else:
                        self.logger.warning(f"unexpected tokenized format: {type(chunk_ids)}")
                return {'input_ids': chunked_input_ids}
            except Exception as e:
                identifier = None
                if isinstance(examples, dict):
                    identifier = examples.get('path') or examples.get('文件名') or examples.get('md5')
                self.logger.error(f"tokenize_and_chunk failed for {identifier or '<unknown>'}: {e}")
                return {'input_ids': []}

        try:
            dataset = dataset.map(
                tokenize_and_chunk,
                batched=True,
                batch_size=100,
                remove_columns=dataset.column_names,
                #remove_columns=['来源', '仓库名', 'path', '文件名', 'ext', 'size', '原始编码', 'md5', 'text', '时间'], # keys are known
            )
            dataset = dataset.filter(
                lambda x: isinstance(x, dict) and isinstance(x.get('input_ids', None), list) and len(x['input_ids']) > 0
            )
        except Exception as e:
            self.logger.error(f"failed transforming dataset pipeline: {e}")
            return iter([])

        return iter(dataset) # return the pointer