#!/usr/bin/env python3
"""以人类可读形式逐项展示 JSON 文件中的每个数据元素。

用法:
    python json_reader.py [json文件路径]

默认读取当前目录下的 data.json。展示支持：
    - 上下左右方向键滚动（类似 less）
    - Enter 进入下一项
    - b 返回上一项，q 或 Esc 退出
    - PgUp / PgDn 翻页，Home / End 跳到首尾
"""
import argparse
import curses
import json
import sys
import unicodedata

# 折行后额外缩进的空格数（悬挂缩进）
HANG_INDENT = 6

# 不宜出现在行首的中文标点（折行时尽量不让它们开头）
NO_START = "，。、；：！？）】」』》〉”’…—～·"


def format_value(value):
    """将值转为适合阅读的字符串。"""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def iter_elements(data):
    """将顶层数据规约为可逐项展示的元素序列。"""
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        yield from data.values()
    else:
        yield data


def display_width(text):
    """计算字符串的终端显示列宽（中文等全角字符按 2 列计）。"""
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        for ch in text
    )


def wrap_text(text, max_width):
    """按显示列宽对文本做贪心折行，返回若干行。"""
    if max_width <= 0:
        max_width = 1
    lines = []
    cur = []
    cur_width = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        # 当前字符是不宜开头的标点时，不在此处换行
        if cur and cur_width + cw > max_width and ch not in NO_START:
            lines.append("".join(cur))
            cur = [ch]
            cur_width = cw
        else:
            cur.append(ch)
            cur_width += cw
    if cur:
        lines.append("".join(cur))
    return lines


def build_lines(element, width):
    """把一个数据元素渲染为若干显示行。

    每行是一个分段列表 [(text, is_key), ...]，is_key 为 True 的段
    会被着色。字符串值按段落折行并加悬挂缩进；\n 与 \n\n 均作为
    段落分隔（\n\n 会保留一个空行）。
    """
    lines = []

    def append_pair(key_text, value):
        key_text = str(key_text)
        base_indent = display_width(key_text) + 2
        val_text = format_value(value)

        if isinstance(value, str):
            # key 单独占一行（着色），内容另起一行，与 key 对齐
            lines.append([(f"{key_text}:", True)])
            for raw in val_text.split("\n"):
                paragraph = raw.rstrip()
                if paragraph == "":
                    lines.append([("", False)])  # 空行
                    continue
                wrapped = wrap_text(paragraph, width)
                for i, chunk in enumerate(wrapped):
                    if i == 0:
                        # 段落首行：与 key 对齐（不缩进）
                        lines.append([(chunk, False)])
                    else:
                        # 折行：悬挂缩进
                        lines.append([(" " * HANG_INDENT + chunk, False)])
        else:
            # 嵌套结构：保留原缩进，仅对续行做基础对齐
            val_lines = val_text.split("\n")
            lines.append([(key_text, True), (f": {val_lines[0]}", False)])
            for extra in val_lines[1:]:
                lines.append([(" " * base_indent + extra, False)])

    if isinstance(element, dict):
        for key, value in element.items():
            append_pair(key, value)
    elif isinstance(element, (list, tuple)):
        for i, value in enumerate(element):
            append_pair(i, value)
    else:
        lines.append([(format_value(element), False)])
    return lines


def line_len(segments):
    """计算一行的字符总数（用于水平滚动的边界判断）。"""
    return sum(len(text) for text, _ in segments)


def clamp(value, lo, hi):
    return max(lo, min(value, hi))


def draw_line(screen, row, segments, hoff, width, key_attr):
    """按水平偏移绘制一行（整段写入，避免宽字符错位）。"""
    x = 0
    skip = hoff
    for text, is_key in segments:
        if skip:
            if len(text) <= skip:
                skip -= len(text)
                continue
            text = text[skip:]
            skip = 0
        if x >= width:
            break
        chunk = text[: width - x]
        if chunk:
            try:
                screen.addstr(row, x, chunk, key_attr if is_key else 0)
            except curses.error:
                pass
            x += len(chunk)


def draw_screen(screen, lines, top, hoff, index, total, key_attr, header_attr):
    """根据滚动位置绘制当前元素。"""
    screen.clear()
    height, width = screen.getmaxyx()
    height = max(height, 1)
    width = max(width, 1)

    header = (
        f"[{index}/{total}]  上/下/左/右 滚动 | Enter 下一项 | "
        f"b 上一项 | q 退出"
    )
    try:
        screen.addstr(0, 0, header[: width - 1], header_attr)
    except curses.error:
        pass

    visible = height - 1
    for y_offset, segments in enumerate(lines[top : top + visible]):
        draw_line(screen, 1 + y_offset, segments, hoff, width, key_attr)


def run_viewer(stdscr, elements):
    total = len(elements)
    if total == 0:
        print("没有可展示的数据。")
        return

    # 初始化颜色
    curses.curs_set(0)
    try:
        curses.start_color()
        curses.use_default_colors()
    except curses.error:
        pass
    try:
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        key_attr = curses.color_pair(1) | curses.A_BOLD
        header_attr = curses.color_pair(2) | curses.A_BOLD
    except curses.error:
        key_attr = curses.A_BOLD
        header_attr = curses.A_BOLD

    index = 0
    top = 0
    hoff = 0
    lines = []
    last_width = -1

    while True:
        height, width = stdscr.getmaxyx()
        if width != last_width or not lines:
            lines = build_lines(elements[index], width)
            last_width = width
        visible = max(height - 1, 1)
        max_top = max(0, len(lines) - visible)
        longest = max((line_len(segments) for segments in lines), default=0)
        max_hoff = max(0, longest - width)
        top = clamp(top, 0, max_top)
        hoff = clamp(hoff, 0, max_hoff)

        draw_screen(stdscr, lines, top, hoff, index + 1, total, key_attr, header_attr)
        stdscr.refresh()

        key = stdscr.getch()

        if key in (curses.KEY_UP, ord("k")):
            top -= 1
        elif key in (curses.KEY_DOWN, ord("j")):
            top += 1
        elif key in (curses.KEY_LEFT, ord("h")):
            hoff -= 2
        elif key in (curses.KEY_RIGHT, ord("l")):
            hoff += 2
        elif key in (curses.KEY_PPAGE, ord(" ")):
            top -= visible
        elif key in (curses.KEY_NPAGE,):
            top += visible
        elif key == curses.KEY_HOME:
            top = 0
        elif key == curses.KEY_END:
            top = len(lines) - 1
        elif key in (curses.KEY_ENTER, 10, 13):
            if index + 1 < total:
                index += 1
                top = 0
                hoff = 0
                lines = []
            else:
                break
        elif key in (ord("b"), curses.KEY_BACKSPACE, 127):
            if index > 0:
                index -= 1
                top = 0
                hoff = 0
                lines = []
        elif key in (ord("q"), 27):  # q 或 Esc
            break


def main():
    parser = argparse.ArgumentParser(description="逐项展示 JSON 文件内容")
    parser.add_argument(
        "path",
        nargs="?",
        default="data.json",
        help="要读取的 JSON 文件地址（默认: data.json）",
    )
    args = parser.parse_args()

    try:
        with open(args.path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件不存在: {args.path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    elements = list(iter_elements(data))
    curses.wrapper(run_viewer, elements)


if __name__ == "__main__":
    main()
