from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

class QADataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        path="/data1/dataset/jssnu005/sys/I_Wonder_Why-Chinese/general",
        max_length=768,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("parquet", data_files=f"{path}/*.parquet", split="train")
        # self.dataset = self._filter_valid_samples()
        # As for this dataset, I have filtered it before, and it's all valid.

    def _clean_text(self, value):
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.strip()

    def _filter_valid_samples(self):
        valid_samples = []
        for sample in tqdm(self.dataset):
            prompt = sample.get("prompt")
            response = sample.get("response")
            try:
                self._encode_chat(prompt, response)
            except ValueError:
                print(f"preview: there is an invalid sample in dataset, skip")
                continue
            except Exception as e:
                print(f"preview: there is an unknown invalid sample in dataset, skip. Deatil:\n {e}")
                continue
            valid_samples.append(sample)
        return valid_samples

    def _encode_chat(self, prompt, response):
        prompt = self._clean_text(prompt)
        response = self._clean_text(response)

        full_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        full = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
        )

        if isinstance(full, list):  # 版本差异
            full_ids = full
        else:
            full_ids = full["input_ids"]

        full_ids = list(full_ids)

        assistant_marker = "<|im_start|>assistant"
        marker_tokens = self.tokenizer.encode(assistant_marker, add_special_tokens=False)
        
        # 在 full_ids 中查找 marker_tokens 第一次出现的位置
        pos = -1
        marker_len = len(marker_tokens)
        for i in range(len(full_ids) - marker_len + 1):
            if full_ids[i:i+marker_len] == marker_tokens:
                pos = i + marker_len  # 位置的边界处理正确吗？是否需要+1/-1
                break

        if pos == -1:
            raise ValueError("no assistant tag")

        # 截断到 max_length（右截断）
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            if pos >= len(full_ids):
                raise ValueError("assistant is totally deleted")
            pass

        labels = [-100] * len(full_ids)
        assistant_end = len(full_ids)
        labels[pos:len(full_ids)] = full_ids[pos:len(full_ids)]

        pad_len = max(0, self.max_length - len(full_ids))
        full_ids = full_ids + [self.tokenizer.pad_token_id] * pad_len
        labels = labels + [-100] * pad_len
        attention_mask = [1] * (self.max_length - pad_len) + [0] * pad_len

        return full_ids, labels, attention_mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt = sample.get("prompt")
        response = sample.get("response")

        input_ids, labels, attention_mask = self._encode_chat(prompt, response)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    base_path = "/data1/dataset/jssnu005/sys/qwen3-0.6B" # [SMOKE]
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    
    ds = QADataset(tokenizer=tokenizer)
    print(len(ds))

    for x in ds:
        x=x["input_ids"]
        print(tokenizer.decode(x,skip_special_tokens=False))
        input("按回车键继续...")