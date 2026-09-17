"""GPT 中国象棋应用的启动入口。

职责：读取环境配置并组装 OpenAI 客户端、CchessBoard 棋盘、CchessAnalyzer
分析器和 CchessGUI 图形界面，不在本文件中实现棋盘规则或界面逻辑。

主要接口：
    main() -> None
        无参数。创建应用依赖并进入 GUI 的事件循环，直到用户关闭窗口。

调用方式：
    在本目录（``~/practice/Cchess``）执行 ``python main.py``。完整运行需要
    配置 ``DEEPSEEK_API_KEY``；分析器为内置搜索，无需外部引擎。
"""

from openai import OpenAI

try:
    from .modules.analyzer import CchessAnalyzer
    from .modules.board import CchessBoard
    from .modules.config import (
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        SEARCH_DEPTH,
    )
    from .modules.gui import CchessGUI
except ImportError:
    from modules.analyzer import CchessAnalyzer
    from modules.board import CchessBoard
    from modules.config import (
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        SEARCH_DEPTH,
    )
    from modules.gui import CchessGUI


def main():
    if not OPENAI_API_KEY:
        raise SystemExit(
            "缺少必要的环境变量: DEEPSEEK_API_KEY\n"
            "请设置后再启动（DEEPSEEK_API_KEY）。"
        )

    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    analyzer = CchessAnalyzer(SEARCH_DEPTH)
    board = CchessBoard()
    gui = CchessGUI(board, client, analyzer, human_color=1)
    gui.run()


if __name__ == "__main__":
    main()
