"""集中读取 GPT 与局面分析器的运行配置。

职责：从环境变量生成模块级配置常量，不执行网络请求、进程启动或文件
写入。缺少可选配置时，相关常量回退到默认值。

主要输入：
    DEEPSEEK_MODEL：DeepSeek 使用的模型名，默认 ``deepseek-flash``；也兼容
        旧的 OPENAI_MODEL 环境变量。
    DEEPSEEK_BASE_URL：DeepSeek OpenAI 兼容接口地址，默认
        ``https://api.deepseek.com``。
    DEEPSEEK_API_KEY：传给 OpenAI 兼容客户端的 API 密钥。
    CC_SEARCH_DEPTH：内置分析器的搜索深度，默认 2。

主要输出：
    MODEL、OPENAI_BASE_URL、OPENAI_API_KEY、MODEL_OPTIONS、THINKING_OPTIONS、
    SEARCH_DEPTH、DEPTH_OPTIONS 等若干模块级常量。

调用方式：
    ``from modules.config import MODEL``，或在包方式下使用
    ``from Cchess.modules.config import MODEL``。
"""

import os


MODEL = os.getenv("DEEPSEEK_MODEL", os.getenv("OPENAI_MODEL", "deepseek-flash"))
OPENAI_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_OPTIONS = ("deepseek-flash", "deepseek-v4-pro")
THINKING_OPTIONS = ("disabled", "low", "high", "max")

# 中国象棋固定 9 列 × 10 行。
BOARD_COLS = 9
BOARD_ROWS = 10

# 内置分析器搜索深度（层数）与界面可切换的选项。
DEPTH_OPTIONS = (1, 2, 3)
try:
    SEARCH_DEPTH = int(os.getenv("CC_SEARCH_DEPTH", "2"))
except ValueError:
    SEARCH_DEPTH = 2
if SEARCH_DEPTH not in DEPTH_OPTIONS:
    SEARCH_DEPTH = 2
