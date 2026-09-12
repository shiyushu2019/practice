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
    from .config import MODEL, MODEL_OPTIONS, THINKING_OPTIONS
    from .player import ask_model_for_move
except ImportError:
    from modules.analyzer import evaluate_gpt_move
    from modules.config import MODEL, MODEL_OPTIONS, THINKING_OPTIONS
    from modules.player import ask_model_for_move


class GoGUI:
    MIN_WINDOW_WIDTH = 900
    MIN_WINDOW_HEIGHT = 640
    MIN_BOARD_WIDTH = 560

    BG = (222, 184, 135)
    LINE = (0, 0, 0)
    BLACK = (25, 25, 25)
    WHITE = (245, 245, 245)
    RED = (220, 40, 40)
    BLUE = (40, 80, 200)
    GREEN = (30, 140, 60)
    GOLD = (200, 160, 20)
    TEXT = (30, 30, 30)
    DIM = (110, 110, 110)
    BTN = (60, 90, 160)
    BTN_HOVER = (80, 115, 190)
    BTN_DISABLED = (150, 150, 150)
    BTN_DANGER = (170, 70, 70)
    BTN_DANGER_HOVER = (200, 90, 90)
    BTN_ACTIVE = (200, 140, 40)

    def __init__(self, board, client, analyzer, human_color=1):
        pygame.init()
        self.board = board
        self.client = client
        self.analyzer = analyzer
        self.human_color = human_color
        self.ai_color = 3 - human_color
        self.model_options = MODEL_OPTIONS
        self.thinking_options = THINKING_OPTIONS
        self.selected_model = MODEL if MODEL in self.model_options else self.model_options[0]
        self.selected_thinking = "high"
        self.thinking_timeout_text = "30"
        self.thinking_timeout_active = False
        self.thinking_deadline = None
        self.edit_color = 1
        self.reasoning_text = ""
        self.reasoning_window = None
        self.reasoning_renderer = None

        self.cell = 34
        self.margin = 42
        self.panel_gap = 28
        self.panel_width = 260
        self.board_width = self.margin * 2 + (board.size - 1) * self.cell
        width = self.board_width + self.panel_gap + self.panel_width
        height = self.board_width
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("GPT 围棋（KataGo 分析）")

        self.font = self._load_font(15)
        self.small = self._load_font(13)
        self.big = self._load_font(18)
        self._layout_window(width, height)

        self.current_player = 1
        self.last_move_rc = None
        self.last_move_str = None
        self.captured = []
        self.ko_point = None
        self.history = []
        self.move_actors = []
        self.game_over = False
        self.edit_mode = False
        self.ai_thinking = False
        self.analyzing = False
        self.queue = queue.Queue()
        self.kata_analysis = None
        self.gpt_eval = None
        self.snapshots = []
        self.status_lines = []
        self.timeout_popup_text = None
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

        button_width, button_height, gap = self.panel_width, 34, 10
        button_y = max(150, min(190, height // 4))
        self.btn_model = pygame.Rect(self.panel_x, button_y, button_width, button_height)
        self.btn_thinking = pygame.Rect(
            self.panel_x, button_y + button_height + gap, button_width, button_height
        )
        self.btn_timeout = pygame.Rect(
            self.panel_x, button_y + 2 * (button_height + gap), button_width, button_height
        )
        self.btn_edit_color = pygame.Rect(
            self.panel_x, button_y + 3 * (button_height + gap), button_width, button_height
        )
        self.btn_reasoning = pygame.Rect(
            self.panel_x, button_y + 4 * (button_height + gap), button_width, button_height
        )
        action_y = button_y + 5 * (button_height + gap) + 18
        self.btn_analyze = pygame.Rect(self.panel_x, action_y, button_width, button_height)
        self.btn_undo = pygame.Rect(
            self.panel_x, action_y + button_height + gap, button_width, button_height
        )
        self.btn_edit = pygame.Rect(
            self.panel_x, action_y + 2 * (button_height + gap), button_width, button_height
        )
        self.btn_ai = pygame.Rect(
            self.panel_x, action_y + 3 * (button_height + gap), button_width, button_height
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

    def _push_snapshot(self):
        self.snapshots.append({
            "board": [row[:] for row in self.board.board],
            "current_player": self.current_player,
            "ko_point": self.ko_point,
            "last_move_rc": self.last_move_rc,
            "last_move_str": self.last_move_str,
            "captured": list(self.captured),
            "history": list(self.history),
            "move_actors": list(self.move_actors),
            "game_over": self.game_over,
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
        self.history = snapshot["history"]
        self.move_actors = snapshot["move_actors"]
        self.game_over = snapshot["game_over"]
        self.kata_analysis = None
        self.gpt_eval = None
        return True

    def _update_status(self):
        if self.edit_mode:
            status = "编辑模式（左键循环 空→黑→白→空，右键清除）"
        elif self.game_over:
            status = "对局结束"
        elif self.ai_thinking:
            status = "AI 思考中…"
        elif self.analyzing:
            status = "分析中…"
        elif self.current_player == self.human_color:
            status = "你的回合（左键落子 / 右键 pass）"
        else:
            status = "AI 回合"
        lines = [(status, self.big, self.TEXT)]

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
        pygame.draw.rect(
            self.screen, (238, 220, 190),
            (self.board_width, 0, self.panel_gap + self.panel_width, self.board_width),
        )
        for index in range(self.board.size):
            x0, y0 = self._grid_to_pixel(0, index)
            x1, y1 = self._grid_to_pixel(self.board.size - 1, index)
            pygame.draw.line(self.screen, self.LINE, (x0, y0), (x1, y1), 1)
            x0, y0 = self._grid_to_pixel(index, 0)
            x1, y1 = self._grid_to_pixel(index, self.board.size - 1)
            pygame.draw.line(self.screen, self.LINE, (x0, y0), (x1, y1), 1)

        for row in (3, 9, 15):
            for col in (3, 9, 15):
                pygame.draw.circle(self.screen, self.LINE, self._grid_to_pixel(row, col), 3)

        radius = self.cell // 2 - 2
        for row in range(self.board.size):
            for col in range(self.board.size):
                value = self.board.board[row][col]
                if value == 0:
                    continue
                point = self._grid_to_pixel(row, col)
                color = self.BLACK if value == 1 else self.WHITE
                pygame.draw.circle(self.screen, color, point, radius)
                pygame.draw.circle(self.screen, self.LINE, point, radius, 1)

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
                pygame.draw.circle(self.screen, self.GOLD, point, 3)

        if self.last_move_rc is not None:
            pygame.draw.circle(self.screen, self.RED, self._grid_to_pixel(*self.last_move_rc), 3)
        if self.edit_mode:
            pygame.draw.rect(
                self.screen, self.GOLD,
                (self.margin - 14, self.margin - 14,
                 self.grid_span + 28,
                 self.grid_span + 28),
                3, border_radius=6,
            )

        y = 24
        for text, font, color in self.status_lines:
            self.screen.blit(font.render(text, True, color), (self.panel_x, y))
            y += font.get_height() + 3

        mouse = pygame.mouse.get_pos()
        model_label = f"模型：{self.selected_model}（点击切换）"
        thinking_labels = {
            "disabled": "关闭",
            "low": "低",
            "high": "高",
            "max": "最大",
        }
        thinking_label = f"思考：{thinking_labels[self.selected_thinking]}（点击切换）"
        model_color = self.BTN_HOVER if self.btn_model.collidepoint(mouse) else self.BTN
        thinking_color = self.BTN_HOVER if self.btn_thinking.collidepoint(mouse) else self.BTN
        self._draw_button(self.btn_model, model_color, model_label)
        self._draw_button(self.btn_thinking, thinking_color, thinking_label)
        timeout_color = self.BTN_ACTIVE if self.thinking_timeout_active else self.BTN
        self._draw_timeout_input(timeout_color)
        edit_color_label = "黑子" if self.edit_color == 1 else "白子"
        edit_color = self.BTN_HOVER if self.btn_edit_color.collidepoint(mouse) else self.BTN
        self._draw_button(self.btn_edit_color, edit_color, f"编辑落子：{edit_color_label}")
        reasoning_label = "思考链：关闭" if self.reasoning_window else "思考链：打开"
        reasoning_color = self.BTN_HOVER if self.btn_reasoning.collidepoint(mouse) else self.BTN
        self._draw_button(self.btn_reasoning, reasoning_color, reasoning_label)

        if self.analyzing:
            analyze_color, analyze_label = self.BTN_ACTIVE, "分析中…"
        elif self.edit_mode or self.ai_thinking:
            analyze_color, analyze_label = self.BTN_DISABLED, "分析局面"
        else:
            analyze_color = self.BTN_HOVER if self.btn_analyze.collidepoint(mouse) else self.BTN
            analyze_label = "分析局面"
        self._draw_button(self.btn_analyze, analyze_color, analyze_label)

        can_undo = bool(self.snapshots) and not self.ai_thinking and not self.edit_mode and not self.analyzing
        undo_color = self.BTN_HOVER if can_undo and self.btn_undo.collidepoint(mouse) else self.BTN if can_undo else self.BTN_DISABLED
        self._draw_button(self.btn_undo, undo_color, "回退")

        if self.edit_mode:
            edit_color, edit_label = self.BTN_DANGER_HOVER if self.btn_edit.collidepoint(mouse) else self.BTN_DANGER, "退出编辑"
        else:
            edit_color = self.BTN_HOVER if not self.ai_thinking and self.btn_edit.collidepoint(mouse) else self.BTN if not self.ai_thinking else self.BTN_DISABLED
            edit_label = "编辑棋盘"
        self._draw_button(self.btn_edit, edit_color, edit_label)

        can_ai = (not self.ai_thinking and not self.edit_mode and not self.analyzing
                   and not self.game_over and self.current_player == self.ai_color)
        ai_color = self.BTN_HOVER if can_ai and self.btn_ai.collidepoint(mouse) else self.BTN if can_ai else self.BTN_DISABLED
        self._draw_button(self.btn_ai, ai_color, "AI 走棋")
        if self.timeout_popup_text:
            self._draw_timeout_popup()
        pygame.display.flip()

    def _draw_button(self, rect, color, label):
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        text = self.font.render(label, True, (255, 255, 255))
        self.screen.blit(text, (rect.x + (rect.w - text.get_width()) // 2, rect.y + (rect.h - text.get_height()) // 2))

    def _draw_timeout_input(self, color):
        pygame.draw.rect(self.screen, color, self.btn_timeout, border_radius=6)
        border_color = self.WHITE if self.thinking_timeout_active else self.BTN_HOVER
        pygame.draw.rect(self.screen, border_color, self.btn_timeout, 2, border_radius=6)

        if self.ai_thinking and self.thinking_deadline is not None:
            remaining = max(0.0, self.thinking_deadline - time.monotonic())
            value = f"剩余 {remaining:.1f}s"
        else:
            value = self.thinking_timeout_text or "输入秒数"
        label = self.small.render(f"最长思考：{value}", True, self.WHITE)
        label_x = self.btn_timeout.x + 12
        label_y = self.btn_timeout.centery - label.get_height() // 2
        self.screen.blit(label, (label_x, label_y))

        if self.thinking_timeout_active and (pygame.time.get_ticks() // 500) % 2 == 0:
            prefix = self.small.render("最长思考：", True, self.WHITE)
            value_surface = self.small.render(self.thinking_timeout_text, True, self.WHITE)
            cursor_x = label_x + prefix.get_width() + value_surface.get_width()
            pygame.draw.line(
                self.screen,
                self.WHITE,
                (cursor_x, self.btn_timeout.y + 7),
                (cursor_x, self.btn_timeout.bottom - 7),
                2,
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
        if self.timeout_popup_text:
            if (button == 1
                    and hasattr(self, "timeout_popup_button")
                    and self.timeout_popup_button.collidepoint(pos)):
                self.timeout_popup_text = None
            return
        if (button == 1 and self.thinking_timeout_active
                and not self.btn_timeout.collidepoint(pos)):
            self._finish_timeout_edit()
        if button == 1:
            if self.btn_timeout.collidepoint(pos):
                if not self.thinking_timeout_active:
                    self.thinking_timeout_active = True
                    self.thinking_timeout_text = ""
                    pygame.key.start_text_input()
                return
            if self.btn_model.collidepoint(pos):
                if not self.ai_thinking and not self.analyzing:
                    current = self.model_options.index(self.selected_model)
                    self.selected_model = self.model_options[(current + 1) % len(self.model_options)]
                    self._update_status()
                return
            if self.btn_thinking.collidepoint(pos):
                if not self.ai_thinking and not self.analyzing:
                    current = self.thinking_options.index(self.selected_thinking)
                    self.selected_thinking = self.thinking_options[(current + 1) % len(self.thinking_options)]
                    self._update_status()
                return
            if self.btn_edit_color.collidepoint(pos):
                if not self.ai_thinking and not self.analyzing:
                    self.edit_color = 3 - self.edit_color
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

    def _handle_keydown(self, event):
        if not self.thinking_timeout_active:
            return
        if event.key == pygame.K_RETURN:
            self._finish_timeout_edit()
        elif event.key == pygame.K_BACKSPACE:
            self.thinking_timeout_text = self.thinking_timeout_text[:-1]
        elif event.key == pygame.K_ESCAPE:
            self._finish_timeout_edit()
        elif event.unicode and event.unicode in "0123456789.":
            self._append_timeout_text(event.unicode)

    def _finish_timeout_edit(self):
        self.thinking_timeout_active = False
        pygame.key.stop_text_input()
        if not self.thinking_timeout_text:
            self.thinking_timeout_text = "30"

    def _handle_text_input(self, text):
        for char in text:
            self._append_timeout_text(char)

    def _append_timeout_text(self, char):
        if not self.thinking_timeout_active or char not in "0123456789.":
            return
        if char != "." or "." not in self.thinking_timeout_text:
            self.thinking_timeout_text += char

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
            self.board.board[row][col] = self.edit_color
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
        self.thinking_deadline = time.monotonic() + max_thinking_seconds
        self.reasoning_text = ""
        self.timeout_popup_text = None
        if self.reasoning_window:
            self._render_reasoning_window()

        def worker():
            try:
                def on_reasoning(text):
                    self.queue.put(("reasoning_delta", text))

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
                    return_reasoning=True,
                    on_reasoning=on_reasoning,
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
        cols = self.board.cols
        size = self.board.size

        def worker():
            try:
                analyzer.set_board(board_snapshot, cols, size)
                analysis = analyzer.analyze(player)
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
                    self.thinking_deadline = None
                    move = message[1]
                    self.reasoning_text = message[2] or "模型没有返回思考内容。"
                    legal = self.board.legal_moves(self.current_player, self.ko_point)
                    if move.lower() != "pass" and move.lower() not in {item.lower() for item in legal}:
                        move = "pass"
                    self._apply_move(move, actor="ai")
                elif kind == "reasoning_delta":
                    self.reasoning_text += message[1]
                    self._render_reasoning_window()
                elif kind == "thinking_timeout":
                    self.thinking_deadline = time.monotonic()
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
                    self.thinking_deadline = None
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
