"""封装 Stockfish UCI 进程，并比较 GPT 着法与引擎候选着法。

``ChessAnalyzer`` 启动一个 Stockfish UCI 子进程，通过标准输入发送命令，
通过标准输出读取分析结果。局面以 FEN 字符串输入，分析结果以字典输出，
包含最佳着法、评分（厘兵或杀棋步数）、候选列表和当前视角。
``evaluate_gpt_move`` 不启动进程，只接收分析字典和 GPT 着法，返回评分
损失及质量评级。

若系统没有安装 Stockfish，则自动退回到内置的简易子力评估，保证应用
仍然可以完整运行（用于 GPT 对弈的着法由 ``python-chess`` 规则校验）。

主要接口：
    ChessAnalyzer(stockfish_bin=None, multi_pv=4)
        输入 Stockfish 可执行文件路径，返回分析器对象；找不到或启动失败
        时使用内置评估。
    set_position(fen: str) -> None
        把当前局面（FEN）同步到引擎。
    analyze(depth=15) -> dict
        输入搜索深度，返回分析结果字典。
    close() -> None
        关闭 Stockfish 子进程，无返回值。
    evaluate_gpt_move(analysis: dict, gpt_move: str) -> dict | None
        输入分析结果和 GPT 着法，返回评估字典；没有候选结果时返回 ``None``。

调用方式：
    ``analyzer = ChessAnalyzer("/usr/games/stockfish")``，每次分析前调用
    ``set_position(fen)``，再调用 ``analyze(depth)``，程序退出时调用
    ``close()``。
"""

import os
import subprocess
import threading

import chess


class ChessAnalyzer:
    def __init__(self, stockfish_bin=None, multi_pv=4):
        self._bin = stockfish_bin
        self.multi_pv = multi_pv
        self.engine = None
        self._fen = chess.STARTING_FEN
        self._lock = threading.Lock()
        if self._bin and os.path.isfile(self._bin):
            try:
                self._spawn()
            except Exception:
                self.engine = None

    # ------------------------------------------------------------------ #
    # 子进程管理
    # ------------------------------------------------------------------ #
    def _spawn(self):
        self.proc = subprocess.Popen(
            [self._bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send_until("uci", "uciok")
        self._send_only(f"setoption name MultiPV value {self.multi_pv}")
        self._send_only(f"setoption name Threads value {_threads()}")
        self._send_only(f"setoption name Hash value {_hash_size()}")
        self._send_until("isready", "readyok")
        self._send_only("ucinewgame")
        self.engine = True

    def _send_only(self, cmd):
        """发送一条不产生输出的 UCI 命令（如 setoption / position）。"""
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _send_until(self, cmd, stop_prefix):
        """发送一条 UCI 命令并读取输出，直到出现 ``stop_prefix`` 前缀或超时。"""
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        stream = self.proc.stdout
        result = {}

        def reader():
            collected = []
            try:
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    line = line.rstrip("\n")
                    collected.append(line)
                    if line.startswith(stop_prefix):
                        break
                result["lines"] = collected
            except Exception as error:
                result["error"] = error

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(60.0)
        if thread.is_alive():
            raise TimeoutError(f"Stockfish 命令超时: {cmd}")
        if "error" in result:
            raise result["error"]
        lines = result.get("lines", [])
        for line in lines:
            if line.startswith("info string error"):
                raise RuntimeError(f"分析引擎错误: {line}")
        return lines

    def close(self):
        if self.engine is None:
            return
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            self.proc.terminate()
        except Exception:
            pass
        self.engine = None

    # ------------------------------------------------------------------ #
    # 分析
    # ------------------------------------------------------------------ #
    def set_position(self, fen):
        self._fen = fen
        if self.engine is None:
            return
        with self._lock:
            self._send_only(f"position fen {fen}")

    def analyze(self, depth=15):
        if self.engine is None:
            return self._fallback_analyze(self._fen, depth)
        with self._lock:
            lines = self._send_until(f"go depth {int(depth)}", "bestmove")
        return self._parse_search(lines, self._fen, depth)

    @staticmethod
    def _parse_info(line):
        tokens = line.split()
        if not tokens or tokens[0] != "info":
            return None
        data = {"multipv": 1, "depth": 0, "score_cp": None, "score_mate": None, "pv": []}
        index = 1
        while index < len(tokens):
            key = tokens[index]
            if key == "pv":
                data["pv"] = tokens[index + 1:]
                break
            if key in ("multipv", "depth", "seldepth") and index + 1 < len(tokens):
                try:
                    data[key] = int(tokens[index + 1])
                except ValueError:
                    pass
                index += 2
            elif key == "score" and index + 2 < len(tokens):
                score_type = tokens[index + 1]
                try:
                    value = int(tokens[index + 2])
                except ValueError:
                    value = 0
                if score_type == "cp":
                    data["score_cp"] = value
                elif score_type == "mate":
                    data["score_mate"] = value
                index += 3
            else:
                index += 1
        return data

    def _parse_search(self, lines, fen, depth):
        board = chess.Board(fen)
        mover_is_white = board.turn == chess.WHITE
        per_pv = {}
        for line in lines:
            info = self._parse_info(line)
            if not info or not info.get("pv"):
                continue
            per_pv[info["multipv"]] = info

        candidates = []
        for multipv in sorted(per_pv):
            info = per_pv[multipv]
            uci = info["pv"][0]
            move = chess.Move.from_uci(uci)
            san = board.san(move) if move in board.legal_moves else uci
            score_cp = info["score_cp"]
            score_mate = info["score_mate"]
            value = _score_value(score_cp, score_mate, mover_is_white)
            candidates.append({
                "move": san,
                "uci": uci,
                "score_cp": score_cp,
                "score_mate": score_mate,
                "value": value,
                "score_text": _score_text(score_cp, score_mate),
                "depth": info.get("depth", depth),
            })

        if not candidates:
            return {
                "best_move": None,
                "best_value": None,
                "best_score_text": None,
                "candidates": [],
                "view_player": 1 if mover_is_white else 2,
                "engine": "stockfish",
            }
        candidates.sort(key=lambda item: item["value"], reverse=True)
        best = candidates[0]
        return {
            "best_move": best["move"],
            "best_value": best["value"],
            "best_score_text": best["score_text"],
            "candidates": candidates,
            "view_player": 1 if mover_is_white else 2,
            "engine": "stockfish",
        }

    @staticmethod
    def _fallback_analyze(fen, depth):
        board = chess.Board(fen)
        mover_is_white = board.turn == chess.WHITE
        ranked = []
        for move in board.legal_moves:
            board.push(move)
            score_cp = _material_score(board)
            board.pop()
            ranked.append((move, score_cp))
        ranked.sort(
            key=lambda item: item[1] if mover_is_white else -item[1],
            reverse=True,
        )
        candidates = []
        for move, score_cp in ranked[:4]:
            candidates.append({
                "move": board.san(move),
                "uci": move.uci(),
                "score_cp": score_cp,
                "score_mate": None,
                "value": _score_value(score_cp, None, mover_is_white),
                "score_text": _score_text(score_cp, None),
                "depth": 0,
            })
        best = candidates[0]
        return {
            "best_move": best["move"],
            "best_value": best["value"],
            "best_score_text": best["score_text"],
            "candidates": candidates,
            "view_player": 1 if mover_is_white else 2,
            "engine": "material",
        }


# ---------------------------------------------------------------------- #
# 评分工具
# ---------------------------------------------------------------------- #
_PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def _material_score(board):
    """从白方视角计算子力差（厘兵）。"""
    score = 0
    for square, piece in board.piece_map().items():
        value = _PIECE_VALUES[piece.piece_type]
        score += value if piece.color == chess.WHITE else -value
    return score


def _score_value(score_cp, score_mate, mover_is_white):
    """把引擎评分换算成“当前行棋方视角”的分值，便于比较损失。"""
    if score_mate is not None:
        base = 100000 - abs(score_mate) * 100
        return base if score_mate > 0 else -base
    if score_cp is None:
        return 0
    return score_cp if mover_is_white else -score_cp


def _score_text(score_cp, score_mate):
    """把评分换算成从白方视角的可读文本。"""
    if score_mate is not None:
        return f"M{score_mate}" if score_mate > 0 else f"-M{abs(score_mate)}"
    if score_cp is None:
        return "0.00"
    sign = "+" if score_cp >= 0 else ""
    return f"{sign}{score_cp / 100:.2f}"


def _threads():
    try:
        from .config import STOCKFISH_THREADS
        return STOCKFISH_THREADS
    except ImportError:
        return 1


def _hash_size():
    try:
        from .config import STOCKFISH_HASH
        return STOCKFISH_HASH
    except ImportError:
        return 16


def evaluate_gpt_move(analysis, gpt_move):
    """比较 GPT 着法与引擎最佳着法，返回质量评级。

    返回字典包含 ``gpt_move``、``best_move``、``best_score_text``、
    ``gpt_score_text``、``value_loss`` 和 ``quality``；当 GPT 着法不在
    引擎候选列表时，``in_candidates`` 为 ``False``。
    """
    if not analysis or not analysis.get("candidates"):
        return None
    best = analysis["candidates"][0]
    gpt_candidate = next(
        (
            candidate for candidate in analysis["candidates"]
            if candidate["move"] == gpt_move or candidate["uci"] == gpt_move
        ),
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
