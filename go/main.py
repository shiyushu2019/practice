"""GPT 围棋应用的启动入口。

职责：读取环境配置并组装 OpenAI 客户端、GoBoard 棋盘、GoAnalyzer
KataGo 分析器和 GoGUI 图形界面，不在本文件中实现棋盘规则或界面逻辑。

主要接口：
    main() -> None
        无参数。创建应用依赖并进入 GUI 的事件循环，直到用户关闭窗口。

调用方式：
    在本目录执行 ``python main.py``，或从父目录执行
    ``python -m go.main``。完整运行需要配置 DEEPSEEK_API_KEY、
    KATAGO_BIN、KATAGO_MODEL 和 KATAGO_CONFIG 环境变量。
"""

from openai import OpenAI

try:
    from .modules.analyzer import GoAnalyzer
    from .modules.board import GoBoard
    from .modules.config import (
        BOARD_SIZE,
        KATAGO_BIN,
        KATAGO_CONFIG,
        KATAGO_MODEL,
        KOMI,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        RULES,
    )
    from .modules.gui import GoGUI
except ImportError:
    from modules.analyzer import GoAnalyzer
    from modules.board import GoBoard
    from modules.config import (
        BOARD_SIZE,
        KATAGO_BIN,
        KATAGO_CONFIG,
        KATAGO_MODEL,
        KOMI,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        RULES,
    )
    from modules.gui import GoGUI


def main():
    missing = []
    if not OPENAI_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not KATAGO_BIN:
        missing.append("KATAGO_BIN")
    if not KATAGO_MODEL:
        missing.append("KATAGO_MODEL")
    if not KATAGO_CONFIG:
        missing.append("KATAGO_CONFIG")
    if missing:
        raise SystemExit(
            "缺少必要的环境变量: " + ", ".join(missing)
            + "\n请设置后再启动（DEEPSEEK_API_KEY 与 KATAGO_BIN/MODEL/CONFIG）。"
        )

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    analyzer = GoAnalyzer(
        KATAGO_BIN,
        KATAGO_MODEL,
        KATAGO_CONFIG,
        komi=KOMI,
        rules=RULES,
        size=BOARD_SIZE,
    )
    board = GoBoard(BOARD_SIZE)
    gui = GoGUI(board, client, analyzer, human_color=1)
    gui.run()


if __name__ == "__main__":
    main()
