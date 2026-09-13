"""中国象棋局面分析器：内置极小极大搜索 + 着法质量评估。

中国象棋没有像 Stockfish 那样普遍预装的引擎，因此 ``CchessAnalyzer``
使用内置的 alpha-beta 搜索对当前局面做浅层分析，返回最佳着法、评分与
候选着法列表，并比较 GPT 着法与最佳着法给出质量评级。搜索与评估全部
基于 ``board.py`` 的规则实现，无需外部依赖。

主要接口：
    CchessAnalyzer(depth=2)
        输入搜索深度，返回分析器对象。
    set_position(fen: str) -> None
        把当前局面（序列化字符串）同步到分析器。
    analyze(depth=None) -> dict
        返回分析结果字典，包含最佳着法、评分、候选列表与当前视角。
    close() -> None
        无外部资源需要释放，保留接口以与其它版本一致。
    evaluate_gpt_move(analysis: dict, gpt_move: str) -> dict | None
        输入分析结果和 GPT 着法，返回评估字典；没有候选时返回 ``None``。

调用方式：
    ``analyzer = CchessAnalyzer(2)``，每次分析前调用 ``set_position(fen)``，
    再调用 ``analyze(depth)``。
"""

from .board import CchessBoard, RED, BLACK, KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN

MATE = 100000
_INF = 10 ** 9

_PIECE_VALUES = {
    KING: 10000,
    ADVISOR: 200,
    ELEPHANT: 200,
    HORSE: 400,
    ROOK: 900,
    CANNON: 450,
    PAWN: 100,
}
_PAWN_CROSS_BONUS = 50
_MOBILITY_BONUS = 5


class CchessAnalyzer:
    def __init__(self, depth=2):
        self.depth = depth
        self._fen = CchessBoard().serialize()
        self._board = CchessBoard()

    def set_position(self, fen):
        self._fen = fen
        self._board.restore(fen)

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # 评估与搜索
    # ------------------------------------------------------------------ #
    @staticmethod
    def _evaluate_red(board):
        """从红方视角评估局面（分值单位近似“厘兵”）。"""
        score = 0
        for r in range(10):
            for c in range(9):
                piece = board.grid[r][c]
                if piece is None:
                    continue
                value = _PIECE_VALUES[piece[1]]
                if piece[0] == RED:
                    score += value
                    if piece[1] == PAWN and r <= 4:
                        score += _PAWN_CROSS_BONUS
                else:
                    score -= value
                    if piece[1] == PAWN and r >= 5:
                        score -= _PAWN_CROSS_BONUS
        score += _MOBILITY_BONUS * (
            len(board._pseudo_moves(RED)) - len(board._pseudo_moves(BLACK))
        )
        return score

    def _negamax(self, board, depth, color, alpha, beta, ply):
        """color 为当前行棋方视角的符号：红方 +1、黑方 -1。"""
        moves = board.legal_moves_raw(board.turn)
        if not moves:
            return -MATE + ply
        if depth == 0:
            return self._quiescence(board, color, alpha, beta, 8)
        best = -_INF
        for fr, fc, tr, tc in moves:
            captured = board._make_move(fr, fc, tr, tc)
            board.turn = 3 - board.turn
            value = -self._negamax(board, depth - 1, -color, -beta, -alpha, ply + 1)
            board.turn = 3 - board.turn
            board._unmake_move(fr, fc, tr, tc, captured)
            if value > best:
                best = value
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _capture_moves(self, board):
        captures = []
        for fr, fc, tr, tc in board.legal_moves_raw(board.turn):
            if board.grid[tr][tc] is not None:
                captures.append((fr, fc, tr, tc))
        return captures

    def _quiescence(self, board, color, alpha, beta, qdepth):
        """静态搜索：在叶子节点继续展开吃子，消除地平线效应。"""
        stand_pat = color * self._evaluate_red(board)
        if qdepth <= 0:
            return stand_pat
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        for fr, fc, tr, tc in self._capture_moves(board):
            captured = board._make_move(fr, fc, tr, tc)
            board.turn = 3 - board.turn
            value = -self._quiescence(board, -color, -beta, -alpha, qdepth - 1)
            board.turn = 3 - board.turn
            board._unmake_move(fr, fc, tr, tc, captured)
            if value >= beta:
                return beta
            if value > alpha:
                alpha = value
        return alpha

    def analyze(self, depth=None):
        depth = self.depth if depth is None else int(depth)
        board = self._board
        color = 1 if board.turn == RED else -1
        moves = board.legal_moves_raw(board.turn)
        mover_is_red = board.turn == RED

        if not moves:
            return {
                "best_move": None,
                "best_value": -MATE,
                "best_score_text": _score_text(-MATE),
                "candidates": [],
                "view_player": board.turn,
            }

        scored = []
        for fr, fc, tr, tc in moves:
            captured = board._make_move(fr, fc, tr, tc)
            board.turn = 3 - board.turn
            value = -self._negamax(board, depth - 1, -color, -_INF, _INF, 1)
            board.turn = 3 - board.turn
            board._unmake_move(fr, fc, tr, tc, captured)
            scored.append(((fr, fc, tr, tc), value))

        scored.sort(key=lambda item: -item[1])
        candidates = []
        for (fr, fc, tr, tc), value in scored[:4]:
            candidates.append({
                "move": board.move_notation(fr, fc, tr, tc),
                "value": value,
                "score_text": _score_text(value if mover_is_red else -value),
                "depth": depth,
            })

        best = candidates[0]
        return {
            "best_move": best["move"],
            "best_value": best["value"],
            "best_score_text": _score_text(
                scored[0][1] if mover_is_red else -scored[0][1],
            ),
            "candidates": candidates,
            "view_player": RED if mover_is_red else BLACK,
        }


def _score_text(red_value):
    """把红方视角分值换算成可读文本。"""
    if red_value >= MATE - 1000:
        return "红方胜势"
    if red_value <= -MATE + 1000:
        return "黑方胜势"
    return f"{red_value / 100:+.2f}"


def evaluate_gpt_move(analysis, gpt_move):
    """比较 GPT 着法与最佳着法，返回质量评级。"""
    if not analysis or not analysis.get("candidates"):
        return None
    best = analysis["candidates"][0]
    gpt_candidate = next(
        (candidate for candidate in analysis["candidates"]
         if candidate["move"] == gpt_move),
        None,
    )
    result = {
        "gpt_move": gpt_move,
        "best_move": best["move"],
        "best_score_text": best["score_text"],
        "best_value": best["value"],
    }
    if gpt_candidate:
        value_loss = (best["value"] or 0) - (gpt_candidate["value"] or 0)
        quality = _quality(value_loss)
        result.update({
            "gpt_score_text": gpt_candidate["score_text"],
            "gpt_value": gpt_candidate["value"],
            "value_loss": value_loss,
            "quality": quality,
            "in_candidates": True,
        })
    else:
        result.update({
            "gpt_score_text": None,
            "gpt_value": None,
            "value_loss": None,
            "quality": "不在候选",
            "in_candidates": False,
        })
    return result


def _quality(value_loss):
    if value_loss < 30:
        return "好棋"
    if value_loss < 100:
        return "一般"
    if value_loss < 250:
        return "较差"
    return "很差"
