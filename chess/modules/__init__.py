"""GPT 国际象棋应用的功能模块包。

本包集中放置棋盘规则、Stockfish 分析器、模型调用、GUI 和运行配置。
各模块以普通 Python 对象交互：棋盘模块维护状态，分析器和模型模块
读取状态并返回分析结果或着法，GUI 模块负责展示状态和处理用户操作。

棋盘规则由 ``python-chess`` 库提供（``chess.Board``），本包在它之上
再封装一层与围棋项目一致的接口，便于上层复用相同的交互逻辑。

调用方式：
    通过 ``from chess.modules.board import ChessBoard`` 等路径导入具体模块；
    通常由上层入口 ``chess.main`` 统一创建和连接这些对象。
"""
