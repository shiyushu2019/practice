"""负责 GPT 围棋提示词构建、模型请求和返回着法解析。

本模块不修改棋盘状态，只读取 GoBoard 当前局面和合法着法，把它们组装
成系统提示词与用户提示词，然后调用 OpenAI 兼容客户端。模型输出会先
解析为坐标，再与棋盘提供的合法着法比对；多次输出无效时自动回退到
``pass`` 或合法着法列表中的第一项。

主要接口：
    build_user_prompt(board, player, last_move, captured, ko_point,
                      legal_moves, history) -> str
        输入棋盘、玩家、上一手、提子、劫点、合法着法和历史，返回用户提示词。
    parse_move(text: str) -> str | None
        输入模型原始文本，返回规范化坐标（如 ``D4``）或 ``pass``；无法
        解析时返回 ``None``。
    ask_model_for_move(client, board, player, last_move, captured,
                       ko_point, history, model=MODEL,
                       thinking_effort="high", komi=7.5, max_retries=3,
                       return_reasoning=False) -> str | tuple[str, str]
        输入 OpenAI 兼容客户端和当前对局信息，返回一个合法坐标或 ``pass``；
        ``return_reasoning=True`` 时额外返回 DeepSeek 思考文本。

调用方式：
    ``move = ask_model_for_move(client, board, player, last_move, captured,
    ko_point, history)``。客户端必须提供
    ``client.chat.completions.create(...)`` 接口。
"""

import re
import queue
import threading
import time

try:
    from .config import MODEL
except ImportError:
    from modules.config import MODEL


SYSTEM_PROMPT_TEMPLATE = """你是围棋对弈程序中的“选点模块”。棋盘状态、提子、劫、合法着法均由外部程序维护，你不负责重新计算规则。

规则：
- {size} 路围棋，黑先，贴目 {komi}。
- 坐标：列 {first_col}-{last_col}，跳过 I；行 1-{size}，从下到上。
- 只能从每步提供的“合法着法”列表中选择。
- 如果合法着法列表为空，输出 pass。
- 如果局面已明显无胜机，输出 resign 认输。

任务：
根据当前完整棋盘、对局信息和合法着法，选择下一手。

时间效率（非常重要）：
- 你的思考 token 有限，每一手都要尽量简短，通常几句话就能给出选点。
- 绝大多数局面不需要长篇分析：不要反复罗列候选、不要逐点评估、不要复述局面。
- 只有在复杂战斗、死活对杀或局面明显胶着时，才允许稍长一些的分析。
- 给定局面后直接输出最合理的一手，避免过度思考。

内部思考流程（不要输出）：
1. 确认轮到谁、上一手、提子、劫禁着点。
2. 找气少的棋、可提的子、可连接或切断的点。
3. 考虑做眼、破眼、逃孤、占角、守边、先手官子。
4. 生成 3-5 个候选，评估安全性、价值和后续。
5. 从合法着法中选最佳一手。

输出格式：
<move>坐标</move>
例如：<move>D4</move>、<move>pass</move> 或 <move>resign</move>
不要输出解释、标点或多余文字。"""


def build_system_prompt(size, komi):
    cols = "ABCDEFGHJKLMNOPQRST"[:size]
    return SYSTEM_PROMPT_TEMPLATE.format(
        size=size,
        komi=f"{komi:g}",
        first_col=cols[0],
        last_col=cols[-1],
    )


def build_user_prompt(board, player, last_move, captured, ko_point, legal_moves, history, previous_reasoning=None):
    player_name = "黑" if player == 1 else "白"
    captured_str = "无" if not captured else ", ".join(captured)
    ko_str = ko_point if ko_point else "无"
    recent = history[-20:] if history else ["无"]
    memory = ""
    if previous_reasoning:
        memory = (
            "\n【上一手 AI 的思考回顾（仅供参考，不一定要沿用）】\n"
            + previous_reasoning[-800:]
        )
    return f"""【当前局面】
{board.board_text()}

【对局信息】
轮到：{player_name}
上一手：{last_move or "无"}
上一手提子：{captured_str}
劫禁着点：{ko_str}
最近着法：{'; '.join(recent)}
{memory}
【合法着法】
{', '.join(legal_moves)}

【任务】
从合法着法中选择一手。内部思考，最后只输出：
<move>坐标</move>"""


def parse_move(text):
    text = text.strip()
    pattern = r"<move>\s*([A-HJ-T](?:1[0-9]|[1-9])|pass|resign|认输)\s*</move>"
    match = re.search(pattern, text, re.I)
    if match:
        return _normalize_move(match.group(1))
    match = re.search(r"\b([A-HJ-T](?:1[0-9]|[1-9])|pass|resign)\b", text, re.I)
    if match:
        return _normalize_move(match.group(1))
    if "认输" in text:
        return "resign"
    return None


def _normalize_move(move):
    move = move.lower()
    if move in ("resign", "认输"):
        return "resign"
    if move == "pass":
        return "pass"
    return move.upper()


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
    captured,
    ko_point,
    history,
    model=MODEL,
    thinking_effort="high",
    komi=7.5,
    max_retries=3,
    return_reasoning=False,
    on_reasoning=None,
    on_usage=None,
    previous_reasoning=None,
    on_illegal=None,
    max_thinking_seconds=None,
    on_timeout=None,
):
    legal = board.legal_moves(player, ko_point)
    legal_set = {move.lower() for move in legal}
    system_prompt = build_system_prompt(board.size, komi)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(
            board, player, last_move, captured, ko_point, legal, history,
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
                legal,
                reasoning_parts,
                return_reasoning,
                on_timeout,
                on_usage,
            )
        content = "".join(content_parts).strip()
        move = parse_move(content)
        if move == "resign":
            if return_reasoning:
                return move, "\n\n".join(reasoning_parts)
            return move
        if move and (move.lower() == "pass" or move.lower() in legal_set):
            if return_reasoning:
                return move, "\n\n".join(reasoning_parts)
            return move
        if on_illegal:
            on_illegal(move, content, attempt)
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (
                f"你上次输出 {content!r} 不合法或无法解析。"
                f"请只从以下合法着法中选择，严格输出 <move>坐标</move>：\n"
                f"{', '.join(legal)}"
            ),
        })
    move = "pass" if "pass" in legal_set else legal[0]
    if return_reasoning:
        return move, "\n\n".join(reasoning_parts)
    return move


def _stream_chunks(client, request, deadline):
    """在可截止的读取器中消费流，避免网络阻塞超过用户设定时间。

    使用有界队列并配合取消标志，超时后旧请求会尽快停止，避免后台线程
    与内存无限增长。
    """
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
    legal,
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
                "请基于当前局面和上述遗产，立即只输出 <move>坐标</move>。"
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
    move = parse_move(content)
    legal_set = {item.lower() for item in legal}
    if move != "resign" and (not move or (move.lower() != "pass" and move.lower() not in legal_set)):
        move = "pass" if "pass" in legal_set else legal[0]
    if return_reasoning:
        return move, "\n".join(reasoning_parts)
    return move
