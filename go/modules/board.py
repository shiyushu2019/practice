"""维护围棋棋盘状态并提供基础规则运算。

棋盘使用二维列表保存状态：``0`` 表示空位，``1`` 表示黑棋，``2``
表示白棋；坐标采用围棋习惯的列字母加行号，例如 ``D4``，列字母跳过
``I``。模块负责坐标转换、相邻点遍历、气和棋块计算、提子、劫点、
禁入点判断以及合法着法生成。

主要接口：
    GoBoard(size=19)
        输入棋盘边长，返回一个空棋盘对象。
    coord_to_rc(move: str) -> tuple[int, int]
        输入坐标字符串，返回内部的 ``(row, col)`` 下标。
    rc_to_coord(row: int, col: int) -> str
        输入内部下标，返回棋盘坐标字符串。
    is_legal(move: str, player: int, ko_point: str | None) -> bool
        判断指定玩家的着法是否合法。
    legal_moves(player: int, ko_point: str | None) -> list[str]
        返回所有合法坐标，并始终包含 ``pass``。
    play(move: str, player: int, ko_point: str | None) -> tuple[list[str], str | None]
        执行落子，返回被提棋子坐标列表和新的劫点；非法着法抛出
        ``ValueError``。
    board_text() -> str
        返回可直接放入提示词或日志的棋盘文本。

调用方式：
    ``board = GoBoard(19)``，然后使用 ``board.legal_moves(...)`` 检查着法，
    使用 ``board.play(...)`` 更新状态。
"""


class GoBoard:
    def __init__(self, size=19):
        self.size = size
        self.cols = "ABCDEFGHJKLMNOPQRST"
        self.board = [[0] * size for _ in range(size)]

    def coord_to_rc(self, move):
        move = move.upper()
        c = self.cols.index(move[0])
        r = self.size - int(move[1:])
        return r, c

    def rc_to_coord(self, r, c):
        return f"{self.cols[c]}{self.size - r}"

    def neighbors(self, r, c):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def _group_and_liberties(self, r, c):
        color = self.board[r][c]
        stack = [(r, c)]
        group = {(r, c)}
        libs = set()
        while stack:
            cr, cc = stack.pop()
            for nr, nc in self.neighbors(cr, cc):
                if self.board[nr][nc] == 0:
                    libs.add((nr, nc))
                elif self.board[nr][nc] == color and (nr, nc) not in group:
                    group.add((nr, nc))
                    stack.append((nr, nc))
        return group, libs

    def _remove_captured(self, r, c, opp):
        captured = []
        for nr, nc in self.neighbors(r, c):
            if self.board[nr][nc] == opp:
                group, libs = self._group_and_liberties(nr, nc)
                if not libs:
                    for gr, gc in group:
                        self.board[gr][gc] = 0
                        captured.append((gr, gc))
        return captured

    def is_legal(self, move, player, ko_point):
        if move.lower() == "pass":
            return True
        r, c = self.coord_to_rc(move)
        if self.board[r][c] != 0:
            return False
        if ko_point and (r, c) == self.coord_to_rc(ko_point):
            return False
        old = [row[:] for row in self.board]
        opp = 3 - player
        self.board[r][c] = player
        self._remove_captured(r, c, opp)
        _, libs = self._group_and_liberties(r, c)
        legal = len(libs) > 0
        self.board = old
        return legal

    def legal_moves(self, player, ko_point):
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    move = self.rc_to_coord(r, c)
                    if self.is_legal(move, player, ko_point):
                        moves.append(move)
        moves.append("pass")
        return moves

    def play(self, move, player, ko_point):
        if move.lower() == "pass":
            return [], None
        if not self.is_legal(move, player, ko_point):
            raise ValueError(f"非法着法: {move}")
        r, c = self.coord_to_rc(move)
        opp = 3 - player
        self.board[r][c] = player
        captured_rc = self._remove_captured(r, c, opp)
        captured = [self.rc_to_coord(rr, cc) for rr, cc in captured_rc]
        new_ko = None
        if len(captured_rc) == 1:
            _, libs = self._group_and_liberties(r, c)
            if len(libs) == 1:
                new_ko = self.rc_to_coord(*captured_rc[0])
        return captured, new_ko

    def board_text(self):
        lines = ["   " + " ".join(self.cols)]
        for r in range(self.size):
            row = []
            for c in range(self.size):
                value = self.board[r][c]
                row.append("X" if value == 1 else "O" if value == 2 else ".")
            lines.append(f"{self.size - r:2d} " + " ".join(row))
        return "\n".join(lines)