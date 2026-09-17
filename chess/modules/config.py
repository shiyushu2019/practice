"""集中读取 GPT 和 Stockfish 的运行配置。

职责：从环境变量生成模块级配置常量，不执行网络请求、进程启动或文件
写入。缺少可选配置时，相关常量为 ``None`` 或回退到默认值，具体组件会
在真正启动时报告配置错误。

主要输入：
    DEEPSEEK_MODEL：DeepSeek 使用的模型名，默认 ``deepseek-flash``；也兼容
        旧的 OPENAI_MODEL 环境变量。
    DEEPSEEK_BASE_URL：DeepSeek OpenAI 兼容接口地址，默认
        ``https://api.deepseek.com``。
    DEEPSEEK_API_KEY：传给 OpenAI 兼容客户端的 API 密钥。
    STOCKFISH_BIN：Stockfish 可执行文件路径；未设置时会在常见位置自动
        查找，仍找不到则退回到内置的简易局面评估。
    STOCKFISH_DEPTH：默认搜索深度，默认 15。
    STOCKFISH_THREADS：Stockfish 使用线程数，默认 1。
    STOCKFISH_HASH：Stockfish 哈希表大小（MB），默认 16。

主要输出：
    MODEL、OPENAI_BASE_URL、OPENAI_API_KEY、MODEL_OPTIONS、THINKING_OPTIONS、
    STOCKFISH_BIN、STOCKFISH_DEPTH、STOCKFISH_THREADS、STOCKFISH_HASH、
    DEPTH_OPTIONS 等若干模块级常量。

调用方式：
    ``from modules.config import STOCKFISH_BIN``，或在包方式下使用
    ``from chess.modules.config import STOCKFISH_BIN``。
"""

import os
import shutil


MODEL = os.getenv("DEEPSEEK_MODEL", os.getenv("OPENAI_MODEL", "deepseek-flash"))
OPENAI_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_OPTIONS = ("deepseek-flash", "deepseek-v4-pro")
THINKING_OPTIONS = ("disabled", "low", "high", "max")

# 国际象棋固定 8x8，不需要棋盘大小配置。
BOARD_SIZE = 8

# 搜索深度选项（供界面循环切换）。
DEPTH_OPTIONS = (10, 15, 20, 25)

# Stockfish 可执行文件：优先使用环境变量，其次在常见位置自动查找。
_CANDIDATE_PATHS = [
    "/usr/games/stockfish",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/stockfish/stockfish",
]

_STOCKFISH_BIN = os.getenv("STOCKFISH_BIN")
if not _STOCKFISH_BIN:
    for _path in _CANDIDATE_PATHS:
        if os.path.isfile(_path):
            _STOCKFISH_BIN = _path
            break
if not _STOCKFISH_BIN:
    _STOCKFISH_BIN = shutil.which("stockfish")

STOCKFISH_BIN = _STOCKFISH_BIN

try:
    STOCKFISH_DEPTH = int(os.getenv("STOCKFISH_DEPTH", "15"))
except ValueError:
    STOCKFISH_DEPTH = 15
if STOCKFISH_DEPTH < 1:
    STOCKFISH_DEPTH = 15

try:
    STOCKFISH_THREADS = max(1, int(os.getenv("STOCKFISH_THREADS", "1")))
except ValueError:
    STOCKFISH_THREADS = 1

try:
    STOCKFISH_HASH = max(1, int(os.getenv("STOCKFISH_HASH", "16")))
except ValueError:
    STOCKFISH_HASH = 16
