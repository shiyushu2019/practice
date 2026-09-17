"""维护中国象棋（Xiangqi）棋盘状态并提供基础规则运算。

本模块自行实现中国象棋规则：9 列 × 10 行棋盘，棋子落在交叉点上。
``0`` 表示空位，棋子用 ``(颜色, 兵种)`` 元组表示：颜色 ``1`` 为红方、
``2`` 为黑方；兵种 ``1-7`` 依次为 帅/将、仕/士、相/象、马、车、炮、兵/卒。
坐标采用 ``列字母 + 行号``，列 ``a-i``（从左到右），行 ``0-9``（从上到下，
0 为黑方底线、9 为红方底线）；着法记作 ``起点+终点``，例如 ``b7e7``。

模块负责坐标转换、兵种走法与吃法（马的蹩腿、象的塞眼、炮的隔山打、
兵过河后的横走）、将帅对脸禁手、将军检测、合法着法生成以及终局判定。

主要接口：
    CchessBoard()
        返回初始局面的棋盘对象，红方先行。
    turn_color() -> int
        返回当前轮到谁：``1`` 红方、``2`` 黑方。
    legal_moves(color=None) -> list[str]
        返回指定方（默认当前方）的全部合法着法。
    is_legal(move: str) -> bool
        判断着法是否合法。
    play(move: str) -> dict
        执行着法并返回该手详情；非法着法抛出 ``ValueError``。
    board_text() -> str
        返回可直接放入提示词或日志的棋盘文本。
    serialize() / restore(fen)
        局面序列化与恢复，用于快照与回退。
    outcome_text() -> str | None
        终局时返回中文结果描述，否则返回 ``None``。

调用方式：
    ``board = CchessBoard()``，然后使用 ``board.legal_moves()`` 检查着法，
    使用 ``board.play(...)`` 更新状态。
"""

KING, ADVISOR, ELEPHANT, HORSE, ROOK, CANNON, PAWN = 1, 2, 3, 4, 5, 6, 7
RED, BLACK = 1, 2

_PIECE_LETTER = {
    KING: "K", ADVISOR: "A", ELEPHANT: "E", HORSE: "H",
    ROOK: "R", CANNON: "C", PAWN: "P",
}
_LETTER_PIECE = {letter: piece for piece, letter in _PIECE_LETTER.items()}

COLS = 9
ROWS = 10

_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
# 马八种走法：(行增量, 列增量, 蹩腿点增量)
_HORSE_MOVES = (
    (2, 1, (1, 0)), (2, -1, (1, 0)), (-2, 1, (-1, 0)), (-2, -1, (-1, 0)),
    (1, 2, (0, 1)), (1, -2, (0, -1)), (-1, 2, (0, 1)), (-1, -2, (0, -1)),
)


def _inside(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS


class CchessBoard:
    def __init__(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]
        self.turn = RED
        self.reset()

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def reset(self):
        self.grid = [[None] * COLS for _ in range(ROWS)]
        self.turn = RED
        back = [ROOK, HORSE, ELEPHANT, ADVISOR, KING, ADVISOR, ELEPHANT, HORSE, ROOK]
        for c, piece in enumerate(back):
            self.grid[0][c] = (BLACK, piece)
            self.grid[9][c] = (RED, piece)
        self.grid[2][1] = (BLACK, CANNON)
        self.grid[2][7] = (BLACK, CANNON)
        self.grid[7][1] = (RED, CANNON)
        self.grid[7][7] = (RED, CANNON)
        for c in (0, 2, 4, 6, 8):
            self.grid[3][c] = (BLACK, PAWN)
            self.grid[6][c] = (RED, PAWN)

    # ------------------------------------------------------------------ #
    # 基本查询
    # ------------------------------------------------------------------ #
    def turn_color(self):
        return self.turn

    def piece_at(self, r, c):
        return self.grid[r][c]

    def find_king(self, color):
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.grid[r][c]
                if piece == (color, KING):
                    return r, c
        return None

    # ------------------------------------------------------------------ #
    # 规则辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _in_palace(r, c, color):
        if not (3 <= c <= 5):
            return False
        return 0 <= r <= 2 if color == BLACK else 7 <= r <= 9

    @staticmethod
    def _on_own_side(r, color):
        return r <= 4 if color == BLACK else r >= 5

    def _kings_facing(self):
        red_king = self.find_king(RED)
        black_king = self.find_king(BLACK)
        if red_king is None or black_king is None:
            return False
        r1, c1 = red_king
        r2, c2 = black_king
        if c1 != c2:
            return False
        lo, hi = (r1, r2) if r1 < r2 else (r2, r1)
        for r in range(lo + 1, hi):
            if self.grid[r][c1] is not None:
                return False
        return True

    def _count_between(self, fr, fc, tr, tc):
        """返回两点之间（不含端点）的棋子数；要求同行或同列。"""
        count = 0
        if fr == tr:
            step = 1 if tc > fc else -1
            for c in range(fc + step, tc, step):
                if self.grid[fr][c] is not None:
                    count += 1
        elif fc == tc:
            step = 1 if tr > fr else -1
            for r in range(fr + step, tr, step):
                if self.grid[r][fc] is not None:
                    count += 1
        return count

    def _piece_attacks(self, fr, fc, tr, tc, ptype):
        """判断 (fr,fc) 处的某兵种是否攻击 (tr,tc)。"""
        dr, dc = tr - fr, tc - fc
        if dr == 0 and dc == 0:
            return False
        if ptype == KING:
            return abs(dr) + abs(dc) == 1
        if ptype == ADVISOR:
            return abs(dr) == 1 and abs(dc) == 1
        if ptype == ELEPHANT:
            if not (abs(dr) == 2 and abs(dc) == 2):
                return False
            if not self._on_own_side(tr, RED if self.grid[fr][fc][0] == RED else BLACK):
                return False
            return self.grid[fr + dr // 2][fc + dc // 2] is None
        if ptype == HORSE:
            for mr, mc, (lr, lc) in _HORSE_MOVES:
                if dr == mr and dc == mc:
                    return self.grid[fr + lr][fc + lc] is None
            return False
        if ptype == ROOK:
            if dr == 0 or dc == 0:
                return self._count_between(fr, fc, tr, tc) == 0
            return False
        if ptype == CANNON:
            if dr == 0 or dc == 0:
                return self._count_between(fr, fc, tr, tc) == 1
            return False
        if ptype == PAWN:
            color = self.grid[fr][fc][0]
            if color == RED:
                forward = -1
                crossed = fr <= 4
            else:
                forward = 1
                crossed = fr >= 5
            if dr == forward and dc == 0:
                return True
            if crossed and dr == 0 and abs(dc) == 1:
                return True
            return False
        return False

    def _is_attacked(self, r, c, by_color):
        for rr in range(ROWS):
            for cc in range(COLS):
                piece = self.grid[rr][cc]
                if piece is None or piece[0] != by_color:
                    continue
                if self._piece_attacks(rr, cc, r, c, piece[1]):
                    return True
        return False

    def _is_in_check(self, color):
        if self._kings_facing():
            return True
        king = self.find_king(color)
        if king is None:
            return False
        return self._is_attacked(king[0], king[1], 3 - color)

    # ------------------------------------------------------------------ #
    # 着法生成
    # ------------------------------------------------------------------ #
    def _pseudo_moves(self, color):
        """生成伪着法（不含自将校验），返回 (fr,fc,tr,tc) 列表。"""
        moves = []
        forward = -1 if color == RED else 1
        crossed = lambda r: (r <= 4) if color == RED else (r >= 5)
        for fr in range(ROWS):
            for fc in range(COLS):
                piece = self.grid[fr][fc]
                if piece is None or piece[0] != color:
                    continue
                ptype = piece[1]

                def push(tr, tc):
                    target = self.grid[tr][tc]
                    if target is None or target[0] != color:
                        moves.append((fr, fc, tr, tc))

                if ptype == KING:
                    for dr, dc in _ORTHO:
                        tr, tc = fr + dr, fc + dc
                        if self._in_palace(tr, tc, color):
                            push(tr, tc)
                elif ptype == ADVISOR:
                    for dr, dc in _DIAG:
                        tr, tc = fr + dr, fc + dc
                        if self._in_palace(tr, tc, color):
                            push(tr, tc)
                elif ptype == ELEPHANT:
                    for dr, dc in _DIAG:
                        tr, tc = fr + 2 * dr, fc + 2 * dc
                        if (not _inside(tr, tc)
                                or not self._on_own_side(tr, color)
                                or self.grid[fr + dr][fc + dc] is not None):
                            continue
                        push(tr, tc)
                elif ptype == HORSE:
                    for dr, dc, (lr, lc) in _HORSE_MOVES:
                        tr, tc = fr + dr, fc + dc
                        if (not _inside(tr, tc)
                                or self.grid[fr + lr][fc + lc] is not None):
                            continue
                        push(tr, tc)
                elif ptype == ROOK:
                    for dr, dc in _ORTHO:
                        tr, tc = fr + dr, fc + dc
                        while _inside(tr, tc):
                            target = self.grid[tr][tc]
                            if target is None:
                                moves.append((fr, fc, tr, tc))
                            else:
                                if target[0] != color:
                                    moves.append((fr, fc, tr, tc))
                                break
                            tr += dr
                            tc += dc
                elif ptype == CANNON:
                    for dr, dc in _ORTHO:
                        jumped = False
                        tr, tc = fr + dr, fc + dc
                        while _inside(tr, tc):
                            target = self.grid[tr][tc]
                            if target is None:
                                if not jumped:
                                    moves.append((fr, fc, tr, tc))
                            else:
                                if not jumped:
                                    jumped = True
                                else:
                                    if target[0] != color:
                                        moves.append((fr, fc, tr, tc))
                                    break
                            tr += dr
                            tc += dc
                elif ptype == PAWN:
                    tr = fr + forward
                    if _inside(tr, fc):
                        push(tr, fc)
                    if crossed(fr):
                        for dc in (-1, 1):
                            tc = fc + dc
                            if _inside(fr, tc):
                                push(fr, tc)
        return moves

    def legal_moves_raw(self, color=None):
        """生成合法着法（排除自将与将帅对脸），返回 (fr,fc,tr,tc) 列表。"""
        color = self.turn if color is None else color
        legal = []
        for fr, fc, tr, tc in self._pseudo_moves(color):
            captured = self._make_move(fr, fc, tr, tc)
            if not self._is_in_check(color):
                legal.append((fr, fc, tr, tc))
            self._unmake_move(fr, fc, tr, tc, captured)
        return legal

    def legal_moves(self, color=None):
        color = self.turn if color is None else color
        return sorted(self.move_notation(*move) for move in self.legal_moves_raw(color))

    def is_legal(self, move):
        parsed = self._parse_move(move)
        if parsed is None:
            return False
        fr, fc, tr, tc = parsed
        return (fr, fc, tr, tc) in self.legal_moves_raw(self.turn)

    def _make_move(self, fr, fc, tr, tc):
        captured = self.grid[tr][tc]
        self.grid[tr][tc] = self.grid[fr][fc]
        self.grid[fr][fc] = None
        return captured

    def _unmake_move(self, fr, fc, tr, tc, captured):
        self.grid[fr][fc] = self.grid[tr][tc]
        self.grid[tr][tc] = captured

    def play(self, move):
        parsed = self._parse_move(move)
        if parsed is None:
            raise ValueError(f"非法着法: {move}")
        fr, fc, tr, tc = parsed
        if (fr, fc, tr, tc) not in self.legal_moves_raw(self.turn):
            raise ValueError(f"非法着法: {move}")
        captured = self._make_move(fr, fc, tr, tc)
        self.turn = 3 - self.turn
        return {
            "move": self.move_notation(fr, fc, tr, tc),
            "from": (fr, fc),
            "to": (tr, tc),
            "captured": captured,
            "moved_piece": self.grid[tr][tc],
        }

    def undo(self):
        raise NotImplementedError("请使用 serialize/restore 快照回退")

    # ------------------------------------------------------------------ #
    # 记法与文本
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coord(r, c):
        return f"{chr(ord('a') + c)}{r}"

    def move_notation(self, fr, fc, tr, tc):
        return self._coord(fr, fc) + self._coord(tr, tc)

    def _parse_move(self, move):
        if not isinstance(move, str):
            return None
        token = move.strip().lower()
        if len(token) != 4:
            return None
        try:
            fc = ord(token[0]) - ord("a")
            fr = int(token[1])
            tc = ord(token[2]) - ord("a")
            tr = int(token[3])
        except (ValueError, TypeError):
            return None
        if not (_inside(fr, fc) and _inside(tr, tc)):
            return None
        return fr, fc, tr, tc

    def is_game_over(self):
        return not self.legal_moves_raw(self.turn)

    def is_checkmate(self):
        return self.is_game_over() and self._is_in_check(self.turn)

    def is_stalemate(self):
        return self.is_game_over() and not self._is_in_check(self.turn)

    def is_check(self):
        return self._is_in_check(self.turn)

    def outcome_text(self):
        if not self.is_game_over():
            return None
        loser = self.turn
        winner = 3 - loser
        winner_name = "红方" if winner == RED else "黑方"
        if self._is_in_check(loser):
            return f"{winner_name}胜（将杀）"
        return f"{winner_name}胜（困毙）"

    def board_text(self):
        rows = ["   " + " ".join("abcdefghi")]
        for r in range(ROWS):
            cells = []
            for c in range(COLS):
                piece = self.grid[r][c]
                if piece is None:
                    cells.append("·")
                else:
                    letter = _PIECE_LETTER[piece[1]]
                    cells.append(letter if piece[0] == RED else letter.lower())
            rows.append(f"{r}  " + " ".join(cells))
        rows.append("   " + " ".join("abcdefghi"))
        rows.append(
            "大写=红方（K帅 A仕 E相 H马 R车 C炮 P兵），"
            "小写=黑方（k将 a士 e象 h马 r车 c炮 p卒）"
        )
        return "\n".join(rows)

    def serialize(self):
        rows = []
        for r in range(ROWS):
            row = ""
            empty = 0
            for c in range(COLS):
                piece = self.grid[r][c]
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    letter = _PIECE_LETTER[piece[1]]
                    row += letter if piece[0] == RED else letter.lower()
            if empty:
                row += str(empty)
            rows.append(row)
        side = "r" if self.turn == RED else "b"
        return "/".join(rows) + " " + side

    def restore(self, fen):
        board_part, _, side = fen.partition(" ")
        self.grid = [[None] * COLS for _ in range(ROWS)]
        for r, row in enumerate(board_part.split("/")):
            c = 0
            for char in row:
                if char.isdigit():
                    c += int(char)
                else:
                    color = RED if char.isupper() else BLACK
                    ptype = _LETTER_PIECE[char.upper()]
                    self.grid[r][c] = (color, ptype)
                    c += 1
        self.turn = RED if side.strip().lower() == "r" else BLACK
