"""实现 pygame 图形界面、用户交互和后台任务状态管理。

``GoGUI`` 连接 GoBoard、OpenAI 兼容客户端和 GoAnalyzer：主线程负责绘制
棋盘、按钮和状态文字，后台线程负责 GPT 请求及 KataGo 分析，结果通过
线程安全队列回到主循环。模块会维护当前玩家、历史、劫点、回退快照、
分析结果和对局结束状态。

主要接口：
    GoGUI(board: GoBoard, client, analyzer: GoAnalyzer, human_color=1)
        输入棋盘对象、模型客户端、KataGo 分析器和人类执棋颜色，创建窗口
        并初始化对局状态；``human_color`` 使用 ``1`` 表示黑、``2`` 表示白。
    run() -> None
        启动 pygame 事件循环，持续处理鼠标操作、后台结果和界面刷新，直到
        收到关闭窗口事件；退出时关闭分析器并调用 ``pygame.quit()``。

调用方式：
    ``gui = GoGUI(board, client, analyzer, human_color=1)``，然后调用
    ``gui.run()``。通常由 ``main.py`` 创建依赖并启动，不建议单独实例化。
"""

import os
import queue
import threading
import time

import pygame
from pygame._sdl2 import video

try:
    from .analyzer import evaluate_gpt_move
    from .board import GoBoard
    from .config import BOARD_SIZES, MODEL, MODEL_OPTIONS, THINKING_OPTIONS
    from .player import ask_model_for_move
except ImportError:
    from modules.analyzer import evaluate_gpt_move
    from modules.board import GoBoard
    from modules.config import BOARD_SIZES, MODEL, MODEL_OPTIONS, THINKING_OPTIONS
    from modules.player import ask_model_for_move


class GoGUI:
    MIN_WINDOW_WIDTH = 900
    MIN_WINDOW_HEIGHT = 640
    MIN_BOARD_WIDTH = 560

    BG = (22, 26, 37)
    PANEL_BG = (31, 36, 50)
    PANEL_BG_2 = (43, 49, 66)
    CARD_BG = (50, 57, 75)

    BOARD_TOP = (236, 200, 142)
    BOARD_BOTTOM = (203, 159, 101)
    BOARD_FRAME = (118, 82, 44)

    LINE = (94, 64, 35)
    BLACK = (24, 24, 28)
    BLACK_HI = (86, 90, 102)
    WHITE = (246, 246, 244)
    WHITE_SHADE = (158, 156, 150)

    RED = (235, 87, 87)
    BLUE = (94, 158, 255)
    GREEN = (56, 180, 127)
    GOLD = (242, 181, 60)

    TEXT = (232, 235, 243)
    DIM = (158, 166, 186)
    MUTED = (118, 125, 145)

    BTN = (62, 86, 158)
    BTN_HOVER = (80, 108, 188)
    BTN_DISABLED = (88, 94, 112)
    BTN_DANGER = (186, 70, 70)
    BTN_DANGER_HOVER = (214, 90, 90)
    BTN_ACTIVE = (212, 150, 46)
    BTN_ACTIVE_HOVER = (232, 168, 58)

    RULES_LABELS = {
        "tromp-taylor": "Tromp-Taylor（数子）",
        "chinese": "中国规则",
        "japanese": "日本规则",
        "korean": "韩国规则",
        "aga": "AGA 规则",
        "new-zealand": "新西兰规则",
    }

    def __init__(self, board, client, analyzer, human_color=1):
        pygame.init()
        self.board = board
        self.client = client
        self.analyzer = analyzer
        self.human_color = human_color
        self.ai_color = 3 - human_color
        self.model_options = MODEL_OPTIONS
        self.thinking_options = THINKING_OPTIONS
        self.board_sizes = BOARD_SIZES
        self.komi = analyzer.komi
        self.rules = analyzer.rules
        self.selected_model = MODEL if MODEL in self.model_options else self.model_options[0]
        self.selected_thinking = "high"
        self.thinking_timeout_text = "30"
        self.thinking_timeout_active = False
        self.timeout_edit_text = None
        self.thinking_start = None
        self.settings_open = False
        self.visit_options = (100, 300, 1000, 3000)
        self.selected_visits = 300
        self.memory_enabled = False
        self.last_ai_reasoning = ""
        self.confirm_text = None
        self.confirm_action = None
        self.notice_text = None
        self.reasoning_text = ""
        self.reasoning_window = None
        self.reasoning_renderer = None

        self.cell = 34
        self.margin = 42
        self.panel_gap = 28
        self.panel_width = 260
        initial_board = self.margin * 2 + (board.size - 1) * self.cell
        self._layout_window(
            initial_board + self.panel_gap + self.panel_width,
            initial_board,
        )
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("GPT 围棋（KataGo 分析）")

        self.font = self._load_font(15)
        self.small = self._load_font(13)
        self.big = self._load_font(18)

        self.current_player = 1
        self.last_move_rc = None
        self.last_move_str = None
        self.captured = []
        self.captured_count = {1: 0, 2: 0}
        self.ko_point = None
        self.history = []
        self.move_actors = []
        self.game_over = False
        self.game_result = None
        self.edit_mode = False
        self.ai_thinking = False
        self.analyzing = False
        self.queue = queue.Queue()
        self.kata_analysis = None
        self.gpt_eval = None
        self.snapshots = []
        self.status_lines = []
        self.timeout_popup_text = None
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._update_status()

        if self.current_player == self.ai_color:
            self._start_ai_turn()

    def _layout_window(self, width, height):
        width = max(self.MIN_WINDOW_WIDTH, int(width))
        height = max(self.MIN_WINDOW_HEIGHT, int(height))
        self.window_width = width
        self.window_height = height
        self.panel_width = min(320, max(260, width // 4))
        self.panel_gap = min(28, max(20, width // 40))
        available_board = width - self.panel_width - self.panel_gap
        self.board_width = max(self.MIN_BOARD_WIDTH, min(height, available_board))
        self.margin = max(24, min(64, round(self.board_width * 0.06)))
        self.grid_span = max(1, self.board_width - 2 * self.margin)
        self.cell = self.grid_span / max(1, self.board.size - 1)
        self.panel_x = self.board_width + self.panel_gap

        button_width = self.panel_width
        button_h, gap = 32, 6
        section_gap = 12
        header_h, header_gap = 16, 6

        # 分区：操作(4) + 视图(1) + 设置(1)
        block_height = (
            (header_h + header_gap + 4 * button_h + 3 * gap)
            + section_gap
            + (header_h + header_gap + button_h)
            + section_gap
            + button_h
        )
        y = height - block_height - 10

        self.section_action_y = y
        y += header_h + header_gap
        self.btn_analyze = pygame.Rect(self.panel_x, y, button_width, button_h)
        y += button_h + gap
        self.btn_ai = pygame.Rect(self.panel_x, y, button_width, button_h)
        y += button_h + gap
        self.btn_undo = pygame.Rect(self.panel_x, y, button_width, button_h)
        y += button_h + gap
        self.btn_edit = pygame.Rect(self.panel_x, y, button_width, button_h)
        y += button_h + section_gap

        self.section_view_y = y
        y += header_h + header_gap
        self.btn_reasoning = pygame.Rect(self.panel_x, y, button_width, button_h)
        y += button_h + section_gap

        self.btn_settings = pygame.Rect(self.panel_x, y, button_width, button_h)

        self.sections = [
            ("操作", self.section_action_y),
            ("视图", self.section_view_y),
        ]

        self._layout_settings()
        self._build_board_background()

    def _build_board_background(self):
        surface = pygame.Surface((self.board_width, self.board_width))
        top, bottom = self.BOARD_TOP, self.BOARD_BOTTOM
        for y in range(self.board_width):
            ratio = y / max(1, self.board_width - 1)
            color = tuple(
                int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3)
            )
            pygame.draw.line(surface, color, (0, y), (self.board_width, y))
        self._board_bg = surface

    def _layout_settings(self):
        width = min(540, self.window_width - 80)
        height = min(460, self.window_height - 80)
        x = (self.window_width - width) // 2
        y = (self.window_height - height) // 2
        self.settings_rect = pygame.Rect(x, y, width, height)

        row_h, row_gap = 40, 12
        right_x = x + 30 + 130 + 12
        right_w = width - (30 + 130 + 12) - 30
        ry = y + 64
        self.setting_board_btn = pygame.Rect(right_x, ry, right_w, row_h)
        ry += row_h + row_gap
        self.setting_model_btn = pygame.Rect(right_x, ry, right_w, row_h)
        ry += row_h + row_gap
        self.setting_thinking_btn = pygame.Rect(right_x, ry, right_w, row_h)
        ry += row_h + row_gap
        self.setting_timeout_field = pygame.Rect(right_x, ry, right_w, row_h)
        ry += row_h + row_gap
        self.setting_visits_btn = pygame.Rect(right_x, ry, right_w, row_h)
        ry += row_h + row_gap
        self.setting_memory_btn = pygame.Rect(right_x, ry, right_w, row_h)

        close_w = 110
        self.settings_close_btn = pygame.Rect(
            x + width - close_w - 30, y + height - 48, close_w, 34,
        )

    def _resize_window(self, width, height):
        width = max(self.MIN_WINDOW_WIDTH, int(width))
        height = max(self.MIN_WINDOW_HEIGHT, int(height))
        self._layout_window(width, height)

    @staticmethod
    def _load_font(size, bold=False):
        candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    return pygame.font.Font(path, size)
                except Exception:
                    continue
        return pygame.font.SysFont(None, size)

    def _grid_to_pixel(self, row, col):
        if self.board.size == 1:
            return self.margin, self.margin
        return (
            round(self.margin + col * self.grid_span / (self.board.size - 1)),
            round(self.margin + row * self.grid_span / (self.board.size - 1)),
        )

    def _pixel_to_grid(self, x, y):
        col = round((x - self.margin) / self.cell)
        row = round((y - self.margin) / self.cell)
        if 0 <= row < self.board.size and 0 <= col < self.board.size:
            px, py = self._grid_to_pixel(row, col)
            if (px - x) ** 2 + (py - y) ** 2 <= (self.cell * 0.5) ** 2:
                return row, col
        return None

    def _star_points(self):
        size = self.board.size
        if size == 9:
            return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        if size == 13:
            return [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        return [
            (3, 3), (3, 9), (3, 15),
            (9, 3), (9, 9), (9, 15),
            (15, 3), (15, 9), (15, 15),
        ]

    def _push_snapshot(self):
        self.snapshots.append({
            "board": [row[:] for row in self.board.board],
            "current_player": self.current_player,
            "ko_point": self.ko_point,
            "last_move_rc": self.last_move_rc,
            "last_move_str": self.last_move_str,
            "captured": list(self.captured),
            "captured_count": dict(self.captured_count),
            "history": list(self.history),
            "move_actors": list(self.move_actors),
            "game_over": self.game_over,
            "game_result": self.game_result,
        })

    def _pop_snapshot(self):
        if not self.snapshots:
            return False
        snapshot = self.snapshots.pop()
        self.board.board = snapshot["board"]
        self.current_player = snapshot["current_player"]
        self.ko_point = snapshot["ko_point"]
        self.last_move_rc = snapshot["last_move_rc"]
        self.last_move_str = snapshot["last_move_str"]
        self.captured = snapshot["captured"]
        self.captured_count = snapshot["captured_count"]
        self.history = snapshot["history"]
        self.move_actors = snapshot["move_actors"]
        self.game_over = snapshot["game_over"]
        self.game_result = snapshot.get("game_result")
        self.kata_analysis = None
        self.gpt_eval = None
        return True

    def _update_status(self):
        if self.edit_mode:
            status = "编辑模式（左键循环 空→黑→白→空，右键清除）"
        elif self.game_over:
            status = self.game_result or "对局结束"
        elif self.ai_thinking:
            status = "AI 思考中…"
        elif self.analyzing:
            status = "分析中…"
        elif self.current_player == self.human_color:
            status = "你的回合（左键落子 / 右键 pass）"
        else:
            status = "AI 回合"
        lines = [(status, self.big, self.TEXT)]

        rules_label = self.RULES_LABELS.get(self.rules, self.rules)
        lines.append((
            f"{self.board.size} 路 · 贴目 {self.komi:g} · {rules_label} · "
            f"提子 黑{self.captured_count[1]}/白{self.captured_count[2]}",
            self.small, self.DIM,
        ))
        lines.append((
            f"Token 消耗：{self.total_tokens:,}（提示 {self.prompt_tokens:,} / 补全 {self.completion_tokens:,}）",
            self.small, self.DIM,
        ))

        if self.kata_analysis and self.kata_analysis.get("best_winrate") is not None:
            winrate = self.kata_analysis["best_winrate"]
            score_lead = self.kata_analysis.get("best_score_lead") or 0.0
            who = "黑" if self.kata_analysis["view_player"] == 1 else "白"
            sign = "+" if score_lead >= 0 else ""
            lines.append((
                f"形势：{who}方胜率 {winrate * 100:.1f}%，目数领先 {sign}{score_lead:.1f}",
                self.font, self.BLUE,
            ))
        else:
            lines.append(("点击「分析局面」查看形势与 GPT 走法评估", self.small, self.DIM))

        if self.gpt_eval:
            evaluation = self.gpt_eval
            if evaluation["in_candidates"]:
                lines.append((
                    f"GPT {evaluation['gpt_move']}｜最佳 {evaluation['best_move']}｜"
                    f"胜率损 {evaluation['winrate_loss'] * 100:.1f}%｜"
                    f"目损 {evaluation['score_loss']:+.1f}｜{evaluation['quality']}",
                    self.small,
                    self.GREEN if evaluation["quality"] in ("好棋", "一般") else self.RED,
                ))
            else:
                lines.append((
                    f"GPT {evaluation['gpt_move']}｜最佳 {evaluation['best_move']}｜{evaluation['quality']}",
                    self.small, self.RED,
                ))

        candidates = (self.kata_analysis or {}).get("candidates") or []
        if candidates:
            parts = []
            for candidate in candidates[:4]:
                score_lead = candidate.get("scoreLead") or 0.0
                parts.append(f"{candidate['move'].upper()} {candidate['winrate'] * 100:.0f}% {score_lead:+.1f}")
            lines.append(("候选：" + " ｜ ".join(parts), self.small, self.TEXT))
        self.status_lines = lines

    def _draw(self):
        self.screen.fill(self.BG)
        self._draw_board_background()
        self._draw_board()
        self._draw_panel()
        if self.settings_open:
            self._draw_settings()
        if self.timeout_popup_text:
            self._draw_timeout_popup()
        if self.confirm_text:
            self._draw_confirm_popup()
        if self.notice_text:
            self._draw_notice_popup()
        pygame.display.flip()

    def _draw_board_background(self):
        if getattr(self, "_board_bg", None) is None:
            self._build_board_background()
        self.screen.blit(self._board_bg, (0, 0))
        pygame.draw.rect(
            self.screen, self.PANEL_BG,
            (self.board_width, 0, self.panel_gap, self.window_height),
        )
        pygame.draw.rect(
            self.screen, self.PANEL_BG,
            (self.panel_x, 0, self.panel_width, self.window_height),
        )

    def _draw_board(self):
        for index in range(self.board.size):
            x0, y0 = self._grid_to_pixel(0, index)
            x1, y1 = self._grid_to_pixel(self.board.size - 1, index)
            pygame.draw.line(self.screen, self.LINE, (x0, y0), (x1, y1), 1)
            x0, y0 = self._grid_to_pixel(index, 0)
            x1, y1 = self._grid_to_pixel(index, self.board.size - 1)
            pygame.draw.line(self.screen, self.LINE, (x0, y0), (x1, y1), 1)

        for row, col in self._star_points():
            pygame.draw.circle(self.screen, self.LINE, self._grid_to_pixel(row, col), 3)

        radius = max(4, int(self.cell // 2) - 2)
        last_rc = self.last_move_rc
        for row in range(self.board.size):
            for col in range(self.board.size):
                value = self.board.board[row][col]
                if value == 0:
                    continue
                self._draw_stone(
                    self._grid_to_pixel(row, col), value, radius,
                    last=(row, col) == last_rc,
                )

        candidates = (self.kata_analysis or {}).get("candidates") or []
        for index, candidate in enumerate(candidates[:4]):
            move = candidate["move"].upper()
            if move == "PASS":
                continue
            try:
                row, col = self.board.coord_to_rc(move)
            except Exception:
                continue
            point = self._grid_to_pixel(row, col)
            if index == 0:
                pygame.draw.circle(self.screen, self.GREEN, point, radius + 3, 2)
            else:
                pygame.draw.circle(self.screen, self.GOLD, point, 4)

        self._draw_hover_stone(radius)

        if self.edit_mode:
            pygame.draw.rect(
                self.screen, self.GOLD,
                (self.margin - 14, self.margin - 14,
                 self.grid_span + 28, self.grid_span + 28),
                3, border_radius=6,
            )

        self._draw_coordinates()

    def _draw_stone(self, point, color, radius, last=False):
        x, y = point
        if color == 1:
            base, highlight = self.BLACK, self.BLACK_HI
        else:
            base, highlight = self.WHITE, (255, 255, 255)
        pygame.draw.circle(self.screen, self.BOARD_FRAME, (x + 1, y + 2), radius)
        pygame.draw.circle(self.screen, base, (x, y), radius)
        if color == 2:
            pygame.draw.circle(self.screen, self.WHITE_SHADE, (x, y), radius, 1)
        pygame.draw.circle(
            self.screen, highlight,
            (x - radius // 3, y - radius // 3), max(2, radius // 3),
        )
        if last:
            marker = self.WHITE if color == 1 else self.BLACK
            pygame.draw.circle(self.screen, marker, (x, y), max(2, radius // 4))

    def _draw_hover_stone(self, radius):
        if self.edit_mode or self.game_over or self.ai_thinking or self.analyzing:
            return
        if self.current_player != self.human_color:
            return
        x, y = pygame.mouse.get_pos()
        if x >= self.board_width or y >= self.board_width:
            return
        grid = self._pixel_to_grid(x, y)
        if grid is None:
            return
        row, col = grid
        if self.board.board[row][col] != 0:
            return
        move = self.board.rc_to_coord(row, col)
        if not self.board.is_legal(move, self.current_player, self.ko_point):
            return
        point = self._grid_to_pixel(row, col)
        overlay = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        if self.human_color == 1:
            pygame.draw.circle(overlay, (*self.BLACK, 115), (radius, radius), radius)
        else:
            pygame.draw.circle(overlay, (255, 255, 255, 140), (radius, radius), radius)
            pygame.draw.circle(overlay, (40, 40, 40, 170), (radius, radius), radius, 1)
        self.screen.blit(overlay, (point[0] - radius, point[1] - radius))

    def _draw_coordinates(self):
        label_color = (124, 88, 46)
        for col in range(self.board.size):
            x, _ = self._grid_to_pixel(0, col)
            surface = self.small.render(self.board.cols[col], True, label_color)
            self.screen.blit(surface, (x - surface.get_width() // 2, self.margin - 22))
        for row in range(self.board.size):
            _, y = self._grid_to_pixel(row, 0)
            surface = self.small.render(str(self.board.size - row), True, label_color)
            self.screen.blit(surface, (self.margin - 24, y - surface.get_height() // 2))

    def _draw_panel(self):
        mouse = pygame.mouse.get_pos()
        panel_x = self.panel_x

        title = self.big.render("GPT 围棋", True, self.TEXT)
        subtitle = self.small.render("DeepSeek 选点 · KataGo 分析", True, self.DIM)
        self.screen.blit(title, (panel_x + 16, 16))
        self.screen.blit(subtitle, (panel_x + 16, 44))

        card_top = 68
        card_bottom = self.section_action_y - 8
        self._draw_card(panel_x + 8, card_top, self.panel_width - 16, max(12, card_bottom - card_top))
        y = card_top + 10
        overflow = False
        for text, font, color in self.status_lines:
            for piece in self._wrap_text(text, font, self.panel_width - 36):
                if y + font.get_height() > card_bottom - 8:
                    overflow = True
                    break
                self.screen.blit(font.render(piece, True, color), (panel_x + 20, y))
                y += font.get_height() + 4
            if overflow:
                break

        if self.ai_thinking and self.thinking_start is not None:
            elapsed = max(0.0, time.monotonic() - self.thinking_start)
            countdown = self.font.render(f"已思考 {elapsed:.1f}s", True, self.GOLD)
            if y + countdown.get_height() <= card_bottom - 8:
                self.screen.blit(countdown, (panel_x + 20, y))

        for label, section_y in self.sections:
            self._draw_section_header(section_y, label)

        # 操作
        if self.analyzing:
            analyze_color, analyze_label = self.BTN_ACTIVE, "分析中…"
        elif self.edit_mode or self.ai_thinking:
            analyze_color, analyze_label = self.BTN_DISABLED, "分析局面"
        else:
            analyze_color = self._btn_color(self.btn_analyze, mouse, self.GOLD, self.BTN_ACTIVE_HOVER)
            analyze_label = "分析局面"
        self._draw_button(self.btn_analyze, analyze_color, analyze_label)

        can_ai = (not self.ai_thinking and not self.edit_mode and not self.analyzing
                   and not self.game_over and self.current_player == self.ai_color)
        ai_color = (
            self._btn_color(self.btn_ai, mouse, self.GREEN, (90, 204, 154))
            if can_ai else self.BTN_DISABLED
        )
        self._draw_button(self.btn_ai, ai_color, "AI 走棋")

        can_undo = bool(self.snapshots) and not self.ai_thinking and not self.edit_mode and not self.analyzing
        undo_color = (
            self._btn_color(self.btn_undo, mouse, self.BTN, self.BTN_HOVER)
            if can_undo else self.BTN_DISABLED
        )
        self._draw_button(self.btn_undo, undo_color, "回退")

        if self.edit_mode:
            edit_toggle_color = (
                self.BTN_DANGER_HOVER if self.btn_edit.collidepoint(mouse)
                else self.BTN_DANGER
            )
            edit_label = "退出编辑"
        else:
            edit_toggle_color = (
                self._btn_color(self.btn_edit, mouse, self.BTN, self.BTN_HOVER)
                if not self.ai_thinking else self.BTN_DISABLED
            )
            edit_label = "编辑棋盘"
        self._draw_button(self.btn_edit, edit_toggle_color, edit_label)

        # 视图
        reasoning_label = "思考链：开启" if self.reasoning_window else "思考链：关闭"
        self._draw_button(
            self.btn_reasoning,
            self._btn_color(self.btn_reasoning, mouse, self.BTN, self.BTN_HOVER),
            reasoning_label,
        )

        # 设置
        self._draw_button(
            self.btn_settings,
            self._btn_color(self.btn_settings, mouse, self.BTN, self.BTN_HOVER),
            "设置",
        )

    def _draw_section_header(self, y, label):
        x = self.panel_x + 16
        surface = self.small.render(label, True, self.MUTED)
        self.screen.blit(surface, (x, y))
        line_y = y + surface.get_height() // 2
        pygame.draw.line(
            self.screen, self.PANEL_BG_2,
            (x + surface.get_width() + 8, line_y),
            (self.panel_x + self.panel_width - 16, line_y),
        )

    def _btn_color(self, rect, mouse, base, hover):
        return hover if rect.collidepoint(mouse) else base

    @staticmethod
    def _wrap_text(text, font, max_width):
        if font.size(text)[0] <= max_width:
            return [text]
        lines = []
        current = ""
        for char in text:
            if current and font.size(current + char)[0] > max_width:
                lines.append(current)
                current = char
            else:
                current += char
        if current:
            lines.append(current)
        return lines

    def _draw_card(self, x, y, w, h):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.CARD_BG, rect, border_radius=10)
        pygame.draw.rect(self.screen, self.PANEL_BG_2, rect, 1, border_radius=10)

    def _draw_button(self, rect, color, label):
        pygame.draw.rect(self.screen, (16, 19, 28), rect.move(0, 2), border_radius=8)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        top_highlight = tuple(min(255, channel + 26) for channel in color)
        pygame.draw.line(
            self.screen, top_highlight,
            (rect.x + 8, rect.y + 1), (rect.right - 8, rect.y + 1),
        )
        text = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(
            text,
            (rect.x + (rect.w - text.get_width()) // 2,
             rect.centery - text.get_height() // 2),
        )

    def _draw_settings(self):
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        rect = self.settings_rect
        pygame.draw.rect(self.screen, (38, 44, 60), rect, border_radius=12)
        pygame.draw.rect(self.screen, self.PANEL_BG_2, rect, 1, border_radius=12)

        title = self.big.render("设置", True, self.TEXT)
        self.screen.blit(title, (rect.x + 30, rect.y + 22))

        mouse = pygame.mouse.get_pos()
        self._draw_setting_row("棋盘大小", self.setting_board_btn, f"{self.board.size} 路", mouse)
        self._draw_setting_row("AI 模型", self.setting_model_btn, self.selected_model, mouse)
        thinking_labels = {"disabled": "关闭", "low": "低", "high": "高", "max": "最大"}
        self._draw_setting_row(
            "思考深度", self.setting_thinking_btn,
            thinking_labels[self.selected_thinking], mouse,
        )

        label = self.font.render("最长思考（秒）", True, self.DIM)
        self.screen.blit(
            label,
            (rect.x + 30, self.setting_timeout_field.centery - label.get_height() // 2),
        )
        self._draw_settings_input(self.setting_timeout_field)

        self._draw_setting_row(
            "KataGo 计算量", self.setting_visits_btn, f"{self.selected_visits} 次", mouse,
        )
        self._draw_setting_row(
            "跨手记忆", self.setting_memory_btn,
            "开" if self.memory_enabled else "关", mouse,
        )

        close_color = (
            self.BTN_HOVER if self.settings_close_btn.collidepoint(mouse) else self.BTN
        )
        self._draw_button(self.settings_close_btn, close_color, "完成")

    def _draw_setting_row(self, label, btn, value, mouse):
        x = self.settings_rect.x + 30
        surface = self.font.render(label, True, self.DIM)
        self.screen.blit(surface, (x, btn.centery - surface.get_height() // 2))
        color = self.BTN_HOVER if btn.collidepoint(mouse) else self.BTN
        self._draw_button(btn, color, f"{value}（点击切换）")

    def _draw_settings_input(self, field):
        active = self.thinking_timeout_active
        pygame.draw.rect(self.screen, (16, 19, 28), field.move(0, 2), border_radius=8)
        pygame.draw.rect(self.screen, (50, 57, 75), field, border_radius=8)
        border = self.GOLD if active else self.PANEL_BG_2
        pygame.draw.rect(self.screen, border, field, 1, border_radius=8)

        if active and self.timeout_edit_text is not None:
            value = self.timeout_edit_text
        else:
            value = self.thinking_timeout_text
        label = self.font.render(value or "输入秒数", True, self.TEXT)
        self.screen.blit(label, (field.x + 12, field.centery - label.get_height() // 2))
        if active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor_x = field.x + 12 + label.get_width()
            pygame.draw.line(
                self.screen, self.TEXT,
                (cursor_x, field.y + 8), (cursor_x, field.bottom - 8), 2,
            )

    def _draw_timeout_popup(self):
        popup_width = min(520, self.window_width - 80)
        popup_height = 190
        popup = pygame.Rect(
            (self.window_width - popup_width) // 2,
            (self.window_height - popup_height) // 2,
            popup_width,
            popup_height,
        )
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 105))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, (42, 47, 58), popup, border_radius=8)
        pygame.draw.rect(self.screen, self.GOLD, popup, 2, border_radius=8)

        title = self.font.render("AI 思考超时", True, self.WHITE)
        self.screen.blit(title, (popup.x + 24, popup.y + 20))
        message = self.small.render(self.timeout_popup_text, True, (225, 228, 235))
        self.screen.blit(message, (popup.x + 24, popup.y + 62))

        self.timeout_popup_button = pygame.Rect(
            popup.right - 118,
            popup.bottom - 48,
            94,
            32,
        )
        mouse = pygame.mouse.get_pos()
        color = (
            self.BTN_HOVER
            if self.timeout_popup_button.collidepoint(mouse)
            else self.BTN
        )
        self._draw_button(self.timeout_popup_button, color, "知道了")

    def _draw_confirm_popup(self):
        width = min(460, self.window_width - 80)
        height = 160
        box = pygame.Rect(
            (self.window_width - width) // 2,
            (self.window_height - height) // 2,
            width,
            height,
        )
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, (42, 47, 58), box, border_radius=10)
        pygame.draw.rect(self.screen, self.GOLD, box, 2, border_radius=10)

        message = self.font.render(self.confirm_text, True, self.TEXT)
        self.screen.blit(message, (box.x + 24, box.y + 28))

        self.confirm_ok_btn = pygame.Rect(box.right - 210, box.bottom - 54, 90, 34)
        self.confirm_cancel_btn = pygame.Rect(box.right - 110, box.bottom - 54, 90, 34)
        mouse = pygame.mouse.get_pos()
        ok_color = (
            self.BTN_DANGER_HOVER if self.confirm_ok_btn.collidepoint(mouse)
            else self.BTN_DANGER
        )
        cancel_color = (
            self.BTN_HOVER if self.confirm_cancel_btn.collidepoint(mouse)
            else self.BTN
        )
        self._draw_button(self.confirm_ok_btn, ok_color, "确定")
        self._draw_button(self.confirm_cancel_btn, cancel_color, "取消")

    def _draw_notice_popup(self):
        width = min(520, self.window_width - 80)
        height = 170
        box = pygame.Rect(
            (self.window_width - width) // 2,
            (self.window_height - height) // 2,
            width,
            height,
        )
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, (42, 47, 58), box, border_radius=10)
        pygame.draw.rect(self.screen, self.BLUE, box, 2, border_radius=10)

        title = self.font.render("提示", True, self.WHITE)
        self.screen.blit(title, (box.x + 24, box.y + 20))
        y = box.y + 56
        for line in self._wrap_text(self.notice_text, self.small, width - 48):
            if y > box.bottom - 60:
                break
            self.screen.blit(self.small.render(line, True, (225, 228, 235)), (box.x + 24, y))
            y += self.small.get_height() + 4

        self.notice_button = pygame.Rect(box.right - 118, box.bottom - 48, 94, 32)
        mouse = pygame.mouse.get_pos()
        color = self.BTN_HOVER if self.notice_button.collidepoint(mouse) else self.BTN
        self._draw_button(self.notice_button, color, "知道了")

    def _render_reasoning_window(self):
        if self.reasoning_window is None or self.reasoning_renderer is None:
            return
        width, height = self.reasoning_window.size
        surface = pygame.Surface((width, height))
        surface.fill((35, 38, 45))
        title = self.font.render("DeepSeek 思考链（流式）", True, self.WHITE)
        surface.blit(title, (16, 14))

        content_width = width - 32
        lines = []
        for paragraph in self.reasoning_text.splitlines() or [""]:
            current = ""
            for char in paragraph:
                if current and self.small.size(current + char)[0] > content_width:
                    lines.append(current)
                    current = ""
                current += char
            lines.append(current)
        line_height = self.small.get_height() + 4
        max_lines = max(1, (height - 62) // line_height)
        visible_lines = lines[-max_lines:]
        y = 54
        for line in visible_lines:
            surface.blit(self.small.render(line, True, (225, 228, 235)), (16, y))
            y += line_height
        texture = video.Texture.from_surface(self.reasoning_renderer, surface)
        self.reasoning_renderer.draw_color = (35, 38, 45, 255)
        self.reasoning_renderer.clear()
        self.reasoning_renderer.blit(texture)
        self.reasoning_renderer.present()

    def _open_reasoning_window(self):
        if self.reasoning_window is None:
            window_size = (560, 360)
            self.reasoning_window = video.Window(
                "DeepSeek 思考链",
                size=window_size,
                resizable=True,
            )
            self.reasoning_renderer = video.Renderer(self.reasoning_window)
            desktop_sizes = pygame.display.get_desktop_sizes()
            if desktop_sizes:
                desktop_width, desktop_height = desktop_sizes[0]
                right_margin = 24
                bottom_margin = 48
                self.reasoning_window.position = (
                    max(0, desktop_width - window_size[0] - right_margin),
                    max(0, desktop_height - window_size[1] - bottom_margin),
                )
        self.reasoning_window.show()
        self.reasoning_window.focus()
        self._render_reasoning_window()

    def _close_reasoning_window(self):
        if self.reasoning_window is not None:
            self.reasoning_window.destroy()
        self.reasoning_window = None
        self.reasoning_renderer = None

    def _handle_click(self, pos, button):
        if self.notice_text:
            if button == 1 and hasattr(self, "notice_button") and self.notice_button.collidepoint(pos):
                self.notice_text = None
            return
        if self.confirm_text:
            if button == 1:
                if self.confirm_ok_btn.collidepoint(pos):
                    action = self.confirm_action
                    self.confirm_text = None
                    self.confirm_action = None
                    if action:
                        action()
                elif self.confirm_cancel_btn.collidepoint(pos):
                    self.confirm_text = None
                    self.confirm_action = None
            return
        if self.timeout_popup_text:
            if (button == 1
                    and hasattr(self, "timeout_popup_button")
                    and self.timeout_popup_button.collidepoint(pos)):
                self.timeout_popup_text = None
            return
        if self.settings_open:
            self._handle_settings_click(pos, button)
            return
        if button == 1:
            if self.btn_settings.collidepoint(pos):
                self._open_settings()
                return
            if self.btn_reasoning.collidepoint(pos):
                if self.reasoning_window:
                    self._close_reasoning_window()
                else:
                    self._open_reasoning_window()
                return
            if self.btn_analyze.collidepoint(pos):
                if not self.analyzing and not self.ai_thinking and not self.edit_mode:
                    self._start_analysis()
                return
            if self.btn_undo.collidepoint(pos):
                if self.snapshots and not self.ai_thinking and not self.edit_mode and not self.analyzing:
                    self._undo()
                return
            if self.btn_edit.collidepoint(pos):
                if self.ai_thinking:
                    return
                self._exit_edit_mode() if self.edit_mode else self._enter_edit_mode()
                return
            if self.btn_ai.collidepoint(pos):
                if (not self.ai_thinking and not self.edit_mode and not self.analyzing
                        and not self.game_over and self.current_player == self.ai_color):
                    self._start_ai_turn()
                return

        if self.edit_mode:
            self._handle_edit_click(pos, button)
            return
        if self.game_over or self.ai_thinking or self.analyzing or self.current_player != self.human_color:
            return
        if button == 1:
            grid = self._pixel_to_grid(*pos)
            if grid is not None:
                self._try_play(self.board.rc_to_coord(*grid))
        elif button == 3:
            self._try_play("pass")

    def _handle_settings_click(self, pos, button):
        if button != 1:
            return
        if self.setting_timeout_field.collidepoint(pos):
            if not self.thinking_timeout_active:
                self.thinking_timeout_active = True
                self.timeout_edit_text = ""
                pygame.key.start_text_input()
            return
        if self.thinking_timeout_active:
            self._finish_timeout_edit()
        if self.settings_close_btn.collidepoint(pos):
            self._close_settings()
            return
        if self.setting_board_btn.collidepoint(pos):
            self._request_board_switch()
            return
        if self.setting_model_btn.collidepoint(pos):
            self._cycle_model()
            return
        if self.setting_thinking_btn.collidepoint(pos):
            self._cycle_thinking()
            return
        if self.setting_visits_btn.collidepoint(pos):
            self._cycle_visits()
            return
        if self.setting_memory_btn.collidepoint(pos):
            self.memory_enabled = not self.memory_enabled
            return

    def _open_settings(self):
        self.settings_open = True
        self.thinking_timeout_active = False
        self.timeout_edit_text = None
        pygame.key.stop_text_input()

    def _close_settings(self):
        self.settings_open = False
        self.thinking_timeout_active = False
        self.timeout_edit_text = None
        pygame.key.stop_text_input()

    def _cycle_model(self):
        current = self.model_options.index(self.selected_model)
        self.selected_model = self.model_options[(current + 1) % len(self.model_options)]

    def _cycle_thinking(self):
        current = self.thinking_options.index(self.selected_thinking)
        self.selected_thinking = self.thinking_options[(current + 1) % len(self.thinking_options)]

    def _cycle_visits(self):
        current = self.visit_options.index(self.selected_visits)
        self.selected_visits = self.visit_options[(current + 1) % len(self.visit_options)]

    def _request_board_switch(self):
        if self.ai_thinking or self.analyzing:
            return
        if self.edit_mode or self.history:
            self.confirm_text = "切换棋盘大小会重开新局并清空当前局面，确定继续？"
            self.confirm_action = self._switch_board_size
        else:
            self._switch_board_size()

    def _handle_keydown(self, event):
        if event.key == pygame.K_ESCAPE:
            if self.thinking_timeout_active:
                self._cancel_timeout_edit()
            elif self.settings_open:
                self._close_settings()
            return
        if not self.thinking_timeout_active:
            return
        if event.key == pygame.K_RETURN:
            self._finish_timeout_edit()
        elif event.key == pygame.K_BACKSPACE:
            self.timeout_edit_text = (self.timeout_edit_text or "")[:-1]

    def _finish_timeout_edit(self):
        if self.thinking_timeout_active and (self.timeout_edit_text or "").strip():
            self.thinking_timeout_text = self.timeout_edit_text.strip()
        self.thinking_timeout_active = False
        self.timeout_edit_text = None
        pygame.key.stop_text_input()

    def _cancel_timeout_edit(self):
        self.thinking_timeout_active = False
        self.timeout_edit_text = None
        pygame.key.stop_text_input()

    def _handle_text_input(self, text):
        for char in text:
            self._append_timeout_text(char)

    def _append_timeout_text(self, char):
        if not self.thinking_timeout_active or char not in "0123456789.":
            return
        current = self.timeout_edit_text or ""
        if char != "." or "." not in current:
            self.timeout_edit_text = current + char

    def _get_thinking_timeout(self):
        try:
            value = float(self.thinking_timeout_text)
        except ValueError:
            return 30.0
        return value if value > 0 else 30.0

    def _handle_edit_click(self, pos, button):
        grid = self._pixel_to_grid(*pos)
        if grid is None:
            return
        row, col = grid
        if button == 1:
            self.board.board[row][col] = (self.board.board[row][col] + 1) % 3
        elif button == 3:
            self.board.board[row][col] = 0
        self.kata_analysis = None
        self.gpt_eval = None

    def _enter_edit_mode(self):
        if self.edit_mode:
            return
        self._push_snapshot()
        self.edit_mode = True
        self.ko_point = None
        self.last_move_rc = None
        self.last_move_str = None
        self.captured = []
        self.captured_count = {1: 0, 2: 0}
        self.history = []
        self.move_actors = []
        self.game_over = False
        self.kata_analysis = None
        self.gpt_eval = None
        self._update_status()

    def _exit_edit_mode(self):
        if not self.edit_mode:
            return
        self.edit_mode = False
        self.kata_analysis = None
        self.gpt_eval = None
        self._update_status()
        if not self.game_over and self.current_player == self.ai_color:
            self._start_ai_turn()

    def _undo(self):
        if not self.edit_mode and self._pop_snapshot():
            self._update_status()

    def _try_play(self, move):
        legal = self.board.legal_moves(self.current_player, self.ko_point)
        if move.lower() != "pass" and move.lower() not in {item.lower() for item in legal}:
            return
        self._apply_move(move, actor="human")

    def _apply_move(self, move, actor):
        if move.lower() == "resign":
            self._push_snapshot()
            player = self.current_player
            resign_color = "黑" if player == 1 else "白"
            winner = "白" if player == 1 else "黑"
            self.history.append(f"{resign_color} 认输")
            self.move_actors.append(actor)
            self.game_over = True
            self.game_result = f"{winner}方胜（{resign_color}方认输）"
            self.kata_analysis = None
            self.gpt_eval = None
            self._update_status()
            return
        if move.lower() != "pass":
            legal = self.board.legal_moves(self.current_player, self.ko_point)
            if move.lower() not in {item.lower() for item in legal}:
                return
        self._push_snapshot()
        player = self.current_player
        if move.lower() == "pass":
            self.captured = []
            self.ko_point = None
            self.last_move_rc = None
            self.last_move_str = "pass"
            self.history.append(f"{'黑' if player == 1 else '白'} pass")
        else:
            captured, new_ko = self.board.play(move, player, self.ko_point)
            self.captured = captured
            self.captured_count[player] += len(captured)
            self.ko_point = new_ko
            self.last_move_rc = self.board.coord_to_rc(move)
            self.last_move_str = move
            self.history.append(f"{'黑' if player == 1 else '白'} {move}")
        self.move_actors.append(actor)
        self.current_player = 3 - player
        self.kata_analysis = None
        self.gpt_eval = None
        if (len(self.history) >= 2 and self.history[-1].endswith("pass")
                and self.history[-2].endswith("pass")):
            self.game_over = True
        self._update_status()
        if not self.game_over and self.current_player == self.ai_color:
            self._start_ai_turn()

    def _switch_board_size(self):
        if self.ai_thinking or self.analyzing:
            return
        current = self.board.size
        sizes = self.board_sizes
        index = sizes.index(current) if current in sizes else 0
        next_size = sizes[(index + 1) % len(sizes)]
        try:
            self.analyzer.set_boardsize(next_size)
        except Exception:
            pass
        self.board = GoBoard(next_size)
        self._reset_game()
        self._layout_window(self.window_width, self.window_height)
        self._update_status()

    def _reset_game(self):
        self.current_player = 1
        self.last_move_rc = None
        self.last_move_str = None
        self.captured = []
        self.captured_count = {1: 0, 2: 0}
        self.ko_point = None
        self.history = []
        self.move_actors = []
        self.game_over = False
        self.game_result = None
        self.edit_mode = False
        self.ai_thinking = False
        self.analyzing = False
        self.kata_analysis = None
        self.gpt_eval = None
        self.snapshots = []
        self.reasoning_text = ""
        self.timeout_popup_text = None
        self.thinking_timeout_active = False
        self.timeout_edit_text = None
        self.notice_text = None
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_ai_reasoning = ""
        if self.current_player == self.ai_color:
            self._start_ai_turn()

    def _start_ai_turn(self):
        if self.ai_thinking:
            return
        self.ai_thinking = True
        self._update_status()
        board = self.board
        player = self.current_player
        last_move = self.last_move_str
        captured = list(self.captured)
        ko_point = self.ko_point
        history = list(self.history)
        client = self.client
        model = self.selected_model
        thinking_effort = self.selected_thinking
        max_thinking_seconds = self._get_thinking_timeout()
        self.thinking_start = time.monotonic()
        self.reasoning_text = ""
        self.timeout_popup_text = None
        if self.reasoning_window:
            self._render_reasoning_window()
        prev_reasoning = self.last_ai_reasoning if self.memory_enabled else ""

        def worker():
            try:
                def on_reasoning(text):
                    self.queue.put(("reasoning_delta", text))

                def on_usage(usage):
                    self.queue.put(("usage", usage))

                def on_illegal(parsed_move, raw_content, attempt):
                    self.queue.put(("illegal_move", parsed_move, raw_content, attempt))

                def on_timeout():
                    self.queue.put((
                        "thinking_timeout",
                        max_thinking_seconds,
                    ))

                move = ask_model_for_move(
                    client,
                    board,
                    player,
                    last_move,
                    captured,
                    ko_point,
                    history,
                    model=model,
                    thinking_effort=thinking_effort,
                    komi=self.komi,
                    return_reasoning=True,
                    on_reasoning=on_reasoning,
                    on_usage=on_usage,
                    previous_reasoning=prev_reasoning or None,
                    on_illegal=on_illegal,
                    max_thinking_seconds=max_thinking_seconds,
                    on_timeout=on_timeout,
                )
                self.queue.put(("ai_done", move[0], move[1]))
            except Exception as error:
                self.queue.put(("err", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def _start_analysis(self):
        self.analyzing = True
        self._update_status()
        board_snapshot = [row[:] for row in self.board.board]
        player = self.current_player
        last_actor = self.move_actors[-1] if self.move_actors else None
        last_move = self.last_move_str
        analyzer = self.analyzer
        size = self.board.size
        visits = self.selected_visits

        def worker():
            try:
                analyzer.set_board(board_snapshot, size)
                analysis = analyzer.analyze(player, max_visits=visits)
                evaluation = None
                if last_actor == "ai" and last_move and last_move.lower() != "pass":
                    evaluation = evaluate_gpt_move(analysis, last_move)
                self.queue.put(("analysis_done", analysis, evaluation))
            except Exception as error:
                self.queue.put(("analysis_err", str(error)))

        threading.Thread(target=worker, daemon=True).start()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos, event.button)
                elif event.type == pygame.KEYDOWN:
                    self._handle_keydown(event)
                elif event.type == pygame.TEXTINPUT:
                    self._handle_text_input(event.text)
                elif event.type == pygame.WINDOWCLOSE:
                    window_event = getattr(event, "window", None)
                    window_id = getattr(window_event, "windowID", None)
                    if self.reasoning_window and (
                            window_id is None or window_id == self.reasoning_window.id):
                        self._close_reasoning_window()
                elif event.type in (
                        pygame.WINDOWMAXIMIZED,
                        pygame.WINDOWRESTORED,
                        pygame.WINDOWSIZECHANGED,
                ):
                    window_event = getattr(event, "window", None)
                    window_id = getattr(window_event, "windowID", None)
                    if self.reasoning_window and window_id == self.reasoning_window.id:
                        self._render_reasoning_window()
                    else:
                        width, height = pygame.display.get_window_size()
                        self._resize_window(width, height)
            try:
                message = self.queue.get_nowait()
                kind = message[0]
                if kind == "ai_done":
                    self.ai_thinking = False
                    self.thinking_start = None
                    move = message[1]
                    self.reasoning_text = message[2] or "模型没有返回思考内容。"
                    self.last_ai_reasoning = message[2] or ""
                    if move.lower() == "resign":
                        self._apply_move("resign", actor="ai")
                    else:
                        legal = self.board.legal_moves(self.current_player, self.ko_point)
                        if move.lower() != "pass" and move.lower() not in {item.lower() for item in legal}:
                            move = "pass"
                        self._apply_move(move, actor="ai")
                elif kind == "reasoning_delta":
                    self.reasoning_text += message[1]
                    self._render_reasoning_window()
                elif kind == "usage":
                    usage = message[1] or {}
                    self.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)
                    self._update_status()
                elif kind == "illegal_move":
                    parsed, attempt = message[1], message[3]
                    if parsed:
                        self.notice_text = f"GPT 第 {attempt} 次请求了非法着法「{parsed}」，已重新提交会话。"
                    else:
                        self.notice_text = f"GPT 第 {attempt} 次输出了无法解析的内容，已重新提交会话。"
                elif kind == "thinking_timeout":
                    self.timeout_popup_text = (
                        f"AI 思考已超时（{message[1]:g}s），正在使用遗产快速决策…"
                    )
                elif kind == "analysis_done":
                    self.analyzing = False
                    self.kata_analysis = message[1]
                    self.gpt_eval = message[2]
                    self._update_status()
                elif kind == "analysis_err":
                    self.analyzing = False
                    self.status_lines = [(f"分析失败: {message[1]}", self.font, self.RED)]
                elif kind == "err":
                    self.ai_thinking = False
                    self.thinking_start = None
                    self.status_lines = [(
                        f"AI 调用失败，请点击「AI 走棋」重试: {message[1]}",
                        self.font,
                        self.RED,
                    )]
            except queue.Empty:
                pass
            self._draw()
            clock.tick(30)
        self.analyzer.close()
        self._close_reasoning_window()
        pygame.quit()
