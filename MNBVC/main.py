# git clone https://www.modelscope.cn/datasets/liwu/MNBVC.git "/data1/dataset/jssnu005/sys/MNBVC"

from datasets import load_dataset

data_dir = "/data1/dataset/jssnu005/sys/MNBVC"

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

# 加载数据集（流式）
dataset = load_dataset(
    "json",
    data_files=f"{data_dir}/**/*.jsonl.gz",
    split="train",
    streaming=True,
)

# 应用过滤（惰性，不会立即执行）
dataset = dataset.filter(keep_allowed_ext)
#print("total:",len(dataset))

# 测试：迭代前几条，打印 text 预览
for i, sample in enumerate(dataset):
    print(f"样本 {i}:")
    #print(sample.get('text', '')[:300])  # 只打印前300字符，避免刷屏
    print(sample.get('ext', '')[:300])  # 只打印前300字符，避免刷屏
    print("---")
    if i >= 100:   # 只取前5条演示
        break