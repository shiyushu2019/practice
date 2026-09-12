"""集中读取 GPT 和 KataGo 的运行配置。

职责：从环境变量生成模块级配置常量，不执行网络请求、进程启动或文件
写入。缺少可选配置时，相关常量为 ``None``，具体组件会在真正启动时
报告配置错误。

主要输入：
	DEEPSEEK_MODEL：DeepSeek 使用的模型名，默认 ``deepseek-flash``；也兼容
		旧的 OPENAI_MODEL 环境变量。
	DEEPSEEK_BASE_URL：DeepSeek OpenAI 兼容接口地址，默认
		``https://api.deepseek.com``。
	DEEPSEEK_API_KEY：传给 OpenAI 兼容客户端的 API 密钥。
	KATAGO_BIN：KataGo 可执行文件路径。
	KATAGO_MODEL：KataGo 神经网络模型路径。
	KATAGO_CONFIG：KataGo GTP 配置文件路径。

主要输出：
	MODEL、OPENAI_BASE_URL、OPENAI_API_KEY、KATAGO_BIN、KATAGO_MODEL 和 KATAGO_CONFIG
	五个字符串或 ``None`` 的模块级常量。

调用方式：
	``from modules.config import KATAGO_BIN``，或在包方式下使用
	``from go.modules.config import KATAGO_BIN``。
"""

import os


MODEL = os.getenv("DEEPSEEK_MODEL", os.getenv("OPENAI_MODEL", "deepseek-flash"))
OPENAI_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
OPENAI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MODEL_OPTIONS = ("deepseek-flash", "deepseek-v4-pro")
THINKING_OPTIONS = ("disabled", "low", "high", "max")

KATAGO_BIN = os.getenv("KATAGO_BIN")
KATAGO_MODEL = os.getenv("KATAGO_MODEL")
KATAGO_CONFIG = os.getenv("KATAGO_CONFIG")