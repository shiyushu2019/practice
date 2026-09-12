"""封装 KataGo GTP 进程，并比较 GPT 着法与引擎候选着法。

``GoAnalyzer`` 启动一个 KataGo GTP 子进程，通过标准输入发送命令，
通过标准输出读取分析结果。棋盘状态以二维 ``0/1/2`` 数组输入，分析
结果以字典输出，包含最佳着法、胜率、目数领先、候选列表和当前视角。
``evaluate_gpt_move`` 不启动进程，只接收分析字典和 GPT 坐标，返回胜率
损失、目数损失及质量评级。

主要接口：
    GoAnalyzer(katago_bin: str, model_path: str, config_path: str, komi=7.5,
               rules="tromp-taylor", size=19)
        输入 KataGo 可执行文件、模型、配置路径、贴目、规则名和棋盘路数，
        返回已初始化的分析器对象；初始化会发送棋盘大小、贴目和清盘命令。
    set_boardsize(size: int) -> None
        切换 KataGo 的棋盘大小并清盘。
    set_board(board_state: list[list[int]], cols: str, size=None) -> None
        输入当前棋盘数组、列标签和边长，将局面同步到 KataGo。
    analyze(player: int, max_visits=300) -> dict
        输入当前玩家和最大访问次数，返回分析结果字典。
    close() -> None
        关闭 KataGo 子进程，无返回值。
    evaluate_gpt_move(kata_analysis: dict, gpt_move: str) -> dict | None
        输入 KataGo 分析结果和 GPT 坐标，返回评估字典；没有候选结果时
        返回 ``None``。

调用方式：
    ``analyzer = GoAnalyzer(bin_path, model_path, config_path)``，每次分析
    前调用 ``set_board(...)``，再调用 ``analyze(player)``，程序退出时
    调用 ``close()``。
"""

import os
import subprocess
import tempfile


class GoAnalyzer:
    def __init__(self, katago_bin, model_path, config_path, komi=7.5, rules="tromp-taylor", size=19):
        self.komi = komi
        self.rules = rules
        self.size = size
        self.proc = subprocess.Popen(
            [katago_bin, "gtp", "-model", model_path, "-config", config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send(f"boardsize {size}")
        try:
            self._send(f"kata-set-rules {rules}")
        except RuntimeError:
            # 部分 KataGo 版本不支持该扩展命令，忽略并沿用配置文件中的规则。
            pass
        self._send(f"komi {komi}")
        self._send("clear_board")

    def set_boardsize(self, size):
        self.size = size
        self._send(f"boardsize {size}")
        self._send("clear_board")

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            if line == "":
                break
            lines.append(line)
        for line in lines:
            if line.startswith("?"):
                raise RuntimeError(f"分析引擎错误: {line}")
        return lines

    def set_board(self, board_state, cols, size=None):
        """通过临时 SGF 文件同步任意黑白棋盘局面。"""
        size = self.size if size is None else size
        if size != self.size:
            self._send(f"boardsize {size}")
            self.size = size
        stones = {"B": [], "W": []}
        for r in range(size):
            for c in range(size):
                value = board_state[r][c]
                if value == 0:
                    continue
                color = "B" if value == 1 else "W"
                sgf_coord = f"{chr(ord('a') + c)}{chr(ord('a') + r)}"
                stones[color].append(f"[{sgf_coord}]")

        properties = ["GM[1]", "FF[4]", f"SZ[{size}]", f"KM[{self.komi}]"]
        if stones["B"]:
            properties.append("AB" + "".join(stones["B"]))
        if stones["W"]:
            properties.append("AW" + "".join(stones["W"]))
        sgf = "(;" + "".join(properties) + ")\n"

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="ascii", suffix=".sgf", delete=False
            ) as temp_file:
                temp_file.write(sgf)
                temp_path = temp_file.name
            self._send(f"loadsgf {temp_path}")
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @staticmethod
    def _parse_info(line):
        tokens = line.split()
        if len(tokens) < 3 or tokens[0] != "info":
            return None
        data = {}
        index = 1
        while index < len(tokens):
            key = tokens[index]
            if key == "pv":
                data["pv"] = tokens[index + 1:]
                break
            if index + 1 >= len(tokens):
                break
            value = tokens[index + 1]
            try:
                if key == "move":
                    data[key] = value
                elif key in ("visits", "order"):
                    data[key] = int(value)
                else:
                    data[key] = float(value)
            except ValueError:
                data[key] = value
            index += 2
        return data

    def analyze(self, player, max_visits=300):
        color = "B" if player == 1 else "W"
        lines = self._send(f"kata-genmove_analyze {color} {max_visits}")
        candidates = []
        for line in lines:
            if line.startswith("info "):
                candidate = self._parse_info(line)
                if candidate and "move" in candidate and "winrate" in candidate:
                    candidates.append(candidate)
        try:
            self._send("undo")
        except Exception:
            pass

        if not candidates:
            return {
                "best_move": None,
                "best_winrate": None,
                "best_score_lead": None,
                "candidates": [],
                "view_player": player,
            }
        candidates.sort(key=lambda item: item.get("order", 9999))
        best = candidates[0]
        return {
            "best_move": best["move"].upper(),
            "best_winrate": best["winrate"],
            "best_score_lead": best.get("scoreLead"),
            "candidates": candidates,
            "view_player": player,
        }

    def close(self):
        try:
            self._send("quit")
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass


def evaluate_gpt_move(kata_analysis, gpt_move):
    if not kata_analysis or not kata_analysis.get("candidates"):
        return None
    best = kata_analysis["candidates"][0]
    gpt_candidate = next(
        (candidate for candidate in kata_analysis["candidates"]
         if candidate["move"].upper() == gpt_move.upper()),
        None,
    )
    result = {
        "gpt_move": gpt_move,
        "best_move": best["move"].upper(),
        "best_winrate": best["winrate"],
        "best_score_lead": best.get("scoreLead"),
    }
    if gpt_candidate:
        winrate_loss = best["winrate"] - gpt_candidate["winrate"]
        score_loss = (best.get("scoreLead") or 0) - (gpt_candidate.get("scoreLead") or 0)
        if winrate_loss < 0.03:
            quality = "好棋"
        elif winrate_loss < 0.10:
            quality = "一般"
        elif winrate_loss < 0.20:
            quality = "较差"
        else:
            quality = "很差"
        result.update({
            "gpt_winrate": gpt_candidate["winrate"],
            "gpt_score_lead": gpt_candidate.get("scoreLead"),
            "winrate_loss": winrate_loss,
            "score_loss": score_loss,
            "quality": quality,
            "in_candidates": True,
        })
    else:
        result.update({
            "gpt_winrate": None,
            "gpt_score_lead": None,
            "winrate_loss": None,
            "score_loss": None,
            "quality": "不在候选",
            "in_candidates": False,
        })
    return result