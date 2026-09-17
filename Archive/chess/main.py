"""GPT 国际象棋应用的启动入口。

职责：读取环境配置并组装 OpenAI 客户端、ChessBoard 棋盘、ChessAnalyzer
（Stockfish）分析器和 ChessGUI 图形界面，不在本文件中实现棋盘规则或界面
逻辑。

主要接口：
    main() -> None
        无参数。创建应用依赖并进入 GUI 的事件循环，直到用户关闭窗口。

调用方式：
    在本目录（``~/practice/chess``）执行 ``python main.py``。注意本项目
    目录名与国际象棋库 ``chess`` 同名，请务必从本目录内启动，避免被
    ``python-chess`` 库顶层的 ``chess`` 包遮蔽。完整运行需要配置
    ``DEEPSEEK_API_KEY``；``STOCKFISH_BIN`` 可选，未提供时自动退回内置
    简易局面评估。
"""

from openai import OpenAI

try:
    from .modules.analyzer import ChessAnalyzer
    from .modules.board import ChessBoard
    from .modules.config import (
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        STOCKFISH_BIN,
    )
    from .modules.gui import ChessGUI
except ImportError:
    from modules.analyzer import ChessAnalyzer
    from modules.board import ChessBoard
    from modules.config import (
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        STOCKFISH_BIN,
    )
    from modules.gui import ChessGUI


def main():
    if not OPENAI_API_KEY:
        raise SystemExit(
            "缺少必要的环境变量: DEEPSEEK_API_KEY\n"
            "请设置后再启动（DEEPSEEK_API_KEY；STOCKFISH_BIN 可选）。"
        )

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    analyzer = ChessAnalyzer(STOCKFISH_BIN)
    board = ChessBoard()
    gui = ChessGUI(board, client, analyzer, human_color=1)
    gui.run()


if __name__ == "__main__":
    main()
