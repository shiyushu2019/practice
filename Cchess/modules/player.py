"""负责 GPT 中国象棋提示词构建、模型请求和返回着法解析。

本模块不修改棋盘状态，只读取 CchessBoard 当前局面和合法着法，把它们组装
成系统提示词与用户提示词，然后调用 OpenAI 兼容客户端。模型输出会先解析
为着法，再与棋盘提供的合法着法比对；多次输出无效时自动回退到合法着法
列表中的第一项。

主要接口：
    build_user_prompt(board, player, last_move, history,
                      previous_reasoning=None) -> str
        输入棋盘、玩家、上一手和历史，返回用户提示词。
    parse_move(board, text: str) -> str | None
        输入棋盘和模型原始文本，返回规范化着法或 ``resign``；无法解析时
        返回 ``None``。
    ask_model_for_move(client, board, player, last_move, history, ...)
        输入 OpenAI 兼容客户端和当前对局信息，返回一个合法着法；
        ``return_reasoning=True`` 时额外返回 DeepSeek 思考文本。

调用方式：
    ``move = ask_model_for_move(client, board, player, last_move, history)``。
    客户端必须提供 ``client.chat.completions.create(...)`` 接口。
"""

import queue
import re
import threading
import time

try:
    from .config import MODEL
except ImportError:
    from modules.config import MODEL


SYSTEM_PROMPT_TEMPLATE = """你是中国象棋（Xiangqi）对弈程序中的“选点模块”。棋盘状态、合法着法、将军与终局判定均由外部程序维护，你不负责重新计算规则。

规则：
- 中国象棋，9 列 × 10 行，红方先行。
- 列用字母 a-i（从左到右），行用数字 0-9（从上到下，0=黑方底线，9=红方底线）。
- 着法记作“起点+终点”，例如 b7e7（炮从 b7 平到 e7）、h9g7（马 h9 进 g7）。
- 只能从每步提供的“合法着法”列表中选择。
- 如果合法着法列表为空，说明已无合法着法（将杀或困毙），输出 resign。

棋子记号：
- 红方（大写）：K帅 A仕 E相 H马 R车 C炮 P兵
- 黑方（小写）：k将 a士 e象 h马 r车 c炮 p卒

任务：
根据当前完整棋盘、对局信息和合法着法，选择下一手。

时间效率（最重要，务必严格遵守）：
- 思考预算极其有限：简单/开局局面 0-1 句话，普通局面最多 2 句话，复杂局面最多 3 句话；
  超过即为过度思考，必须避免。
- 开局阶段（前 6 回合内）：不要做任何分析，直接按定式走（中炮、屏风马、仙人指路等），
  立即输出最常规的一手，一秒定着。
- 简单局面（没有被将军、没有对攻、子力大致均衡）：直接按子力发展和占中走，
  不要复述局面、不要罗列候选、不要逐点评估。
- 只有被将军、存在吃子/抽将/捉双等战术、或残局需精确计算时，才允许稍长分析。
- 想好立刻输出 <move>着法</move>，任何多余文字都是错误。

内部思考流程（只在复杂局面才走完，简单/开局局面直接跳到第 5 步）：
1. 先判断是不是开局或简单局面——是的话直接按定式出着，不要展开分析。
2. 确认轮到谁、是否被将军、上一手着法。
3. 只在实际存在战术时检查将军、吃子、捉双、抽将、卧槽马、重炮等机会。
4. 最多想 2-3 个候选，挑一个最稳的即可，不必穷举。
5. 输出最合理的一手。

输出格式：
<move>着法</move>
例如：<move>b7e7</move>、<move>h9g7</move> 或 <move>resign</move>
不要输出解释、标点或多余文字。"""


def build_system_prompt():
    return SYSTEM_PROMPT_TEMPLATE


def build_user_prompt(board, player, last_move, history, previous_reasoning=None):
    player_name = "红方" if player == 1 else "黑方"
    legal = board.legal_moves()
    if legal:
        legal_str = ", ".join(legal)
    else:
        legal_str = "（无合法着法）"
    check_str = "是" if board.is_check() else "否"
    recent = history[-16:] if history else ["无"]
    memory = ""
    if previous_reasoning:
        memory = (
            "\n【上一手 AI 的思考回顾（仅供参考，不一定要沿用）】\n"
            + previous_reasoning[-800:]
        )
    opening_hint = ""
    if len(history) < 6:
        opening_hint = "\n【局面提示】处于开局阶段，请直接按定式或经验选择最常规的一手，不要分析、不要罗列候选。"
    return f"""【当前局面】
{board.board_text()}

【对局信息】
轮到：{player_name}
当前是否被将军：{check_str}
上一手：{last_move or "无"}
最近着法：{'; '.join(recent)}
{memory}{opening_hint}
【合法着法】
{legal_str}

【任务】
从合法着法中选择一手。内部思考，最后只输出：
<move>着法</move>"""


_MOVE_PATTERN = re.compile(r"\b([a-i][0-9][a-i][0-9])\b", re.I)


def _extract_candidates(text):
    candidates = []
    for match in re.finditer(r"<move>\s*(.*?)\s*</move>", text, re.I | re.S):
        candidates.append(match.group(1).strip())
    for token in text.replace("<move>", " ").replace("</move>", " ").split():
        token = token.strip("`'\".,;!?()[]{}<>")
        if token:
            candidates.append(token)
    return candidates


def parse_move(board, text):
    """解析模型原文，返回合法着法或 ``resign``，否则返回 ``None``。"""
    for candidate in _extract_candidates(text):
        if candidate.lower() in ("resign", "认输"):
            return "resign"
    for candidate in _extract_candidates(text):
        if board.is_legal(candidate):
            return candidate.lower()
    return None


def _delta_reasoning(delta):
    """从流式增量中提取思考内容，兼容不同 SDK 的字段位置。"""
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        return reasoning
    extra = getattr(delta, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get("reasoning_content") or extra.get("reasoning")
    return None


def _usage_dict(usage):
    """把 OpenAI 兼容的 usage 对象/字典规范化为 dict。"""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
            "completion_tokens": usage.get("completion_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def ask_model_for_move(
    client,
    board,
    player,
    last_move,
    history,
    model=MODEL,
    thinking_effort="high",
    max_retries=3,
    return_reasoning=False,
    on_reasoning=None,
    on_usage=None,
    previous_reasoning=None,
    on_illegal=None,
    max_thinking_seconds=None,
    on_timeout=None,
):
    legal = board.legal_moves()
    system_prompt = build_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(
            board, player, last_move, history,
            previous_reasoning=previous_reasoning,
        )},
    ]
    reasoning_parts = []
    deadline = (
        time.monotonic() + max_thinking_seconds
        if max_thinking_seconds and max_thinking_seconds > 0
        else None
    )
    for attempt in range(1, max_retries + 1):
        request = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if thinking_effort == "disabled":
            request["temperature"] = 0
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        else:
            request["reasoning_effort"] = thinking_effort
            request["extra_body"] = {"thinking": {"type": "enabled"}}
        content_parts = []
        try:
            chunks = _stream_chunks(client, request, deadline)
            for chunk in chunks:
                if not chunk.choices:
                    usage = getattr(chunk, "usage", None)
                    if usage and on_usage:
                        on_usage(_usage_dict(usage))
                    continue
                delta = chunk.choices[0].delta
                reasoning = _delta_reasoning(delta)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if on_reasoning:
                        on_reasoning(reasoning)
                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
        except TimeoutError:
            return _fallback_after_timeout(
                client,
                model,
                system_prompt,
                messages[1]["content"],
                board,
                reasoning_parts,
                return_reasoning,
                on_timeout,
                on_usage,
            )
        content = "".join(content_parts).strip()
        move = parse_move(board, content)
        if move == "resign":
            if return_reasoning:
                return move, "\n\n".join(reasoning_parts)
            return move
        if move:
            if return_reasoning:
                return move, "\n\n".join(reasoning_parts)
            return move
        if on_illegal:
            on_illegal(None, content, attempt)
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                f"你上次输出 {content!r} 无法解析为合法着法。"
                f"请只从以下合法着法中选择，严格输出 <move>着法</move>：\n"
                f"{', '.join(legal) if legal else '（无合法着法）'}"
            ),
        })
    move = legal[0] if legal else "resign"
    if return_reasoning:
        return move, "\n\n".join(reasoning_parts)
    return move


def _stream_chunks(client, request, deadline):
    """在可截止的读取器中消费流，避免网络阻塞超过用户设定时间。"""
    chunks = queue.Queue(maxsize=256)
    cancel = threading.Event()

    def producer():
        try:
            response = client.chat.completions.create(**request)
            for chunk in response:
                if cancel.is_set():
                    return
                try:
                    chunks.put_nowait(("chunk", chunk))
                except queue.Full:
                    return
            try:
                chunks.put_nowait(("done", None))
            except queue.Full:
                pass
        except Exception as error:
            try:
                chunks.put_nowait(("error", error))
            except queue.Full:
                pass

    threading.Thread(target=producer, daemon=True).start()
    while True:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            cancel.set()
            raise TimeoutError("AI 思考超时")
        try:
            kind, value = chunks.get(timeout=remaining)
        except queue.Empty as error:
            cancel.set()
            raise TimeoutError("AI 思考超时") from error
        if kind == "chunk":
            yield value
        elif kind == "done":
            return
        else:
            raise value


def _fallback_after_timeout(
    client,
    model,
    system_prompt,
    user_prompt,
    board,
    reasoning_parts,
    return_reasoning,
    on_timeout,
    on_usage=None,
):
    """超时后用遗留思考发起关闭思考模式的新会话。"""
    if on_timeout:
        on_timeout()
    legacy = "\n".join(reasoning_parts).strip() or "（没有收到可用的思考内容）"
    fallback_messages = [
        {
            "role": "system",
            "content": (
                system_prompt
                + "\n\n这是一次超时后的快速续会话。请继承下面的遗留思考，"
                "不要继续展开分析，立即输出最终合法着法。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{user_prompt}\n\n【上一个会话的思考遗产】\n{legacy}\n\n"
                "请基于当前局面和上述遗产，立即只输出 <move>着法</move>。"
            ),
        },
    ]
    request = {
        "model": model,
        "messages": fallback_messages,
        "stream": True,
        "temperature": 0,
        "stream_options": {"include_usage": True},
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    fallback_deadline = time.monotonic() + 10
    try:
        response = _stream_chunks(client, request, fallback_deadline)
        content_parts = []
        for chunk in response:
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                if usage and on_usage:
                    on_usage(_usage_dict(usage))
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                content_parts.append(content)
        content = "".join(content_parts).strip()
    except TimeoutError:
        content = ""
    move = parse_move(board, content)
    legal = board.legal_moves()
    if move != "resign" and not move:
        move = legal[0] if legal else "resign"
    if return_reasoning:
        return move, "\n".join(reasoning_parts)
    return move
