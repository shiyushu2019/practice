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

        user_assistant = [
            {"role": "user", "content": prompt},
        ]
        full_messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

        prefix = self.tokenizer.apply_chat_template(
            user_assistant,
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
        )
        full = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
        )

        if isinstance(prefix, list):  # 版本差异
            prefix_ids = prefix
            full_ids = full
        else:
            prefix_ids = prefix["input_ids"]
            full_ids = full["input_ids"]

        prefix_ids = list(prefix_ids)
        full_ids = list(full_ids)

        common_prefix_len = 0
        for prefix_token, full_token in zip(prefix_ids, full_ids):
            if prefix_token != full_token:
                break
            common_prefix_len += 1

        assistant_start = common_prefix_len

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            if assistant_start >= len(full_ids):
                raise ValueError(
                    "assistant segment is completely removed by right truncation; skip this sample"
                )
        else:
            pass

        labels = [-100] * len(full_ids)
        assistant_end = len(full_ids)
        labels[assistant_start:assistant_end] = full_ids[assistant_start:assistant_end]

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

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)
    tokenizer.model_max_length = 10000000
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pad_token_id = tokenizer.pad_token_id
    ds = QADataset(tokenizer=tokenizer)
    print(len(ds))

    for x in ds:
        x=x["input_ids"]
        print(tokenizer.decode(x,skip_special_tokens=True))
        input("按回车键继续...")


