"""维护国际象棋棋盘状态并提供基础规则运算。

本模块在 ``python-chess`` 的 ``chess.Board`` 之上封装一层与围棋项目
一致的接口：着法统一使用 SAN（标准代数记法，例如 ``e4``、``Nf3``、
``exd5``、``O-O``、``e8=Q``），玩家编号沿用 ``1`` 表示白方、``2``
表示黑方。模块负责着法合法性判断、执行着法、生成局面文本、终局判定。

主要接口：
    ChessBoard()
        返回一个初始局面（标准开局）的棋盘对象。
    turn_player() -> int
        返回当前轮到谁：``1`` 白、``2`` 黑。
    legal_moves() -> list[str]
        返回当前局面的所有合法 SAN 着法。
    is_legal(move: str) -> bool
        判断 SAN 着法是否合法（对升变等写法做了宽容处理）。
    play(move: str) -> dict
        执行着法并返回该手的详情；非法着法抛出 ``ValueError``。
    board_text() -> str
        返回可直接放入提示词或日志的 Unicode 棋盘文本与 FEN。
    fen() -> str
        返回当前局面 FEN，可用于快照与回退。
    outcome_text() -> str | None
        对局结束时返回中文结果描述，否则返回 ``None``。

调用方式：
    ``board = ChessBoard()``，然后使用 ``board.legal_moves()`` 检查着法，
    使用 ``board.play(...)`` 更新状态。
"""

import chess


WHITE = 1
BLACK = 2

_PIECE_GLYPHS = chess.UNICODE_PIECE_SYMBOLS


class ChessBoard:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board = chess.Board()

    # ------------------------------------------------------------------ #
    # 局面与坐标
    # ------------------------------------------------------------------ #
    def fen(self):
        return self.board.fen()

    def restore(self, fen):
        self.board = chess.Board(fen)

    def turn_player(self):
        return WHITE if self.board.turn == chess.WHITE else BLACK

    def piece_at_square(self, square):
        return self.board.piece_at(square)

    def symbol_at_square(self, square):
        piece = self.board.piece_at(square)
        return piece.symbol() if piece else ""

    def square_name(self, square):
        return chess.SQUARE_NAMES[square]

    def parse_san(self, move):
        return self.board.parse_san(move)

    def san(self, move):
        return self.board.san(move)

    def uci_to_san(self, uci):
        move = chess.Move.from_uci(uci)
        return self.board.san(move)

    # ------------------------------------------------------------------ #
    # 规则
    # ------------------------------------------------------------------ #
    def legal_moves(self):
        return sorted(self.board.san(move) for move in self.board.legal_moves)

    def is_legal(self, move):
        parsed = _parse_san_lenient(self.board, move)
        if parsed is None:
            return False
        return self.board.is_legal(parsed)

    def normalize_san(self, move):
        """把宽容写法规范化成标准 SAN；非法或无法解析时返回 ``None``。"""
        parsed = _parse_san_lenient(self.board, move)
        if parsed is None:
            return None
        return self.board.san(parsed)

    def play(self, move):
        parsed = _parse_san_lenient(self.board, move)
        if parsed is None:
            raise ValueError(f"非法着法: {move}")
        if not self.board.is_legal(parsed):
            raise ValueError(f"非法着法: {move}")
        san = self.board.san(parsed)
        from_square = parsed.from_square
        to_square = parsed.to_square
        captured = self.board.piece_at(to_square)
        is_castling = self.board.is_castling(parsed)
        is_en_passant = self.board.is_en_passant(parsed)
        is_capture = self.board.is_capture(parsed)
        self.board.push(parsed)
        return {
            "san": san,
            "uci": parsed.uci(),
            "from_square": from_square,
            "to_square": to_square,
            "captured": captured.symbol() if captured else None,
            "promotion": parsed.promotion,
            "is_castling": is_castling,
            "is_en_passant": is_en_passant,
            "is_capture": is_capture,
        }

    def undo(self):
        if self.board.move_stack:
            self.board.pop()

    def is_check(self):
        return self.board.is_check()

    def is_checkmate(self):
        return self.board.is_checkmate()

    def is_stalemate(self):
        return self.board.is_stalemate()

    def is_game_over(self):
        return self.board.is_game_over()

    def is_insufficient_material(self):
        return self.board.is_insufficient_material()

    def result(self):
        return self.board.result()

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #
    def board_text(self):
        """生成 Unicode 棋盘文本，白方在下、黑方在上，并附带 FEN。"""
        rows = []
        rows.append("   " + " ".join("abcdefgh"))
        for rank in range(7, -1, -1):
            cells = []
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece is None:
                    cells.append("·")
                else:
                    cells.append(_PIECE_GLYPHS[piece.symbol()])
            rows.append(f"{rank + 1}  " + " ".join(cells))
        rows.append("   " + " ".join("abcdefgh"))
        rows.append("FEN: " + self.board.fen())
        return "\n".join(rows)

    def outcome_text(self):
        """对局结束时返回中文结果描述，否则返回 ``None``。"""
        if not self.board.is_game_over():
            return None
        outcome = self.board.outcome()
        winner = None if outcome is None else outcome.winner
        winner_name = None if winner is None else ("白方" if winner else "黑方")
        if self.board.is_checkmate():
            return f"{winner_name}胜（将杀）"
        if self.board.is_stalemate():
            return "和棋（逼和）"
        if self.board.is_insufficient_material():
            return "和棋（子力不足）"
        if self.board.can_claim_threefold_repetition():
            return "和棋（三次重复局面）"
        if self.board.is_fifty_moves():
            return "和棋（五十回合规则）"
        if self.board.is_seventyfive_moves():
            return "和棋（七十五回合规则）"
        if self.board.is_fivefold_repetition():
            return "和棋（五次重复局面）"
        return "和棋"


def _parse_san_lenient(board, move):
    """宽容地解析 SAN 着法，兼容常见的记法差异。

    处理的情况：``0-0``/``0-0-0`` 写成了数字零、升变缺少 ``=Q`` 后缀、
    多余的空格或标点。解析失败或结果不合法时返回 ``None``。
    """
    if not isinstance(move, str):
        return None
    token = move.strip().strip(".,;!?")
    if not token:
        return None
    normalized = token
    # 王车易位：数字零 -> 字母 O。
    if token in ("0-0", "o-o", "O-O"):
        normalized = "O-O"
    elif token in ("0-0-0", "o-o-o", "O-O-O"):
        normalized = "O-O-O"
    try:
        parsed = board.parse_san(normalized)
        if board.is_legal(parsed):
            return parsed
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
        pass
    # 升变缺少后缀：尝试补 ``=Q``。
    try:
        parsed = board.parse_san(normalized + "=Q")
        if board.is_legal(parsed):
            return parsed
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
        pass
    return None
