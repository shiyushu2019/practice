import os
import time
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")
dimensions=['文体','核心立意','事件与情节','主体与对象','时空与情境','陈述方式','结构节奏','语言风格','审美效果','创作语境']
example_lines = "\n".join([f"{dim}：..." for dim in dimensions])
MAX_RETRIES = 3

def from_natural_to_format(text):

    system_prompt = f"""你是文学作品创作团队的一名成员，担任用户与主笔写手之间的核心桥梁。你的任务是：精准捕捉用户自然语言中的创作意图，将其转述、补全、整理为结构化的格式化要求，确保主笔写手拿到后无需再猜测用户想法，可直接依据你的输出进行创作。

一、分析角度（共{len(dimensions)}个，必须依次全部输出，一个都不能遗漏）：
{', '.join(dimensions)}

二、输出格式：
一共输出{len(dimensions)}行，每行格式为“角度名称：内容”。不要添加任何额外说明、标题或总结，不要在开头写“输出内容为：”之类，也不要在结尾写“以上就是格式化内容”之类，只依次输出这{len(dimensions)}行。

三、输出格式示例（仅示范格式，这里用...代替内容，你必须基于用户指令独立分析）：
{example_lines}
示例中的...仅为占位符，表示该位置应填入具体分析文字，不代表实际输出长度或内容。

四、填写规则：
1. 原提示词中明显含有的角度，必须如实、原样照搬，不得删减、想象或概括（除非原文实在过长）。例如原提示词说“我要写一个散文”，则“文体”行必须写“文体：散文”。
2. 原提示词未指明的字段，不得留空，必须以最不违和的方式自行创作补全。各个角度要自恰、可以成文。补全示例：若原提示词为“请帮我写一篇小说，内容是王小明今天上学迟到，被老师罚站的故事”，则“文体”写“小说”、“事件与情节”写“王小明今天上学迟到，被老师罚站的故事”、“主体与对象”写“王小明”；其余未明确字段（如核心立意）需自行合理创作补全，例如“核心立意：起到教育意义，启发孩子们不要迟到”。
3. 对未指明的字段尽量发散、写长，以达到能给创作者明确指示的程度，该类字段不得少于100字。
4. 用户给出的模糊化表述，你必须具体化，让下游写手可以直接根据你的格式化内容创作，不得直接把用户的“许愿”式指令传递给下游写手。
5. 所有字段的内容必须是确定的，不得出现“稍后确定”“根据实际情况展开”等偷懒说法。公事公办，不得输出与创作指令无关的内容。
6. 你只是客观处理用户的提示词，不要直接面向下游写手提要求。
7. 忽略原提示词中对篇幅的要求，不要因此改变格式化输出的篇幅。忽略原提示词中非创作要求的其他内容。
"""

    user_prompt = f"""请按照系统要求，对下面这个创作指令进行格式化：

{text}"""
    # print(user_prompt) # debug

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-v4-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
                timeout=200,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
            )
            content = response.choices[0].message.content
            if content is None or len(content.strip())==0:
                raise ValueError("API 返回空内容")
            return content + "\n请根据以上内容进行创作\n"
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError, OSError, ValueError) as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait_time = 2 ** (attempt - 1)
            print(f"[warning] 请求失败（第 {attempt} 次/共 {MAX_RETRIES} 次），{wait_time} 秒后重试：{e}")
            time.sleep(wait_time)

    raise RuntimeError(f"请求失败，已重试 {MAX_RETRIES} 次: {last_error}") if last_error else RuntimeError("请求失败")

if __name__ == "__main__":
    import shutil
    import unicodedata

    # 展示用悬挂缩进的空格数
    HANG_INDENT = 6
    # 不宜出现在行首的中文标点
    NO_START = "，。、；：！？）】」』》〉”’…—～·"

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

    def format_with_hanging_indent(text, width=None):
        """把“角度：内容”多行文本按终端宽度折行并加悬挂缩进。"""
        if width is None:
            width = shutil.get_terminal_size().columns or 80
        result = []
        for line in text.split("\n"):
            if not line.strip():
                result.append("")
                continue
            # 找到第一个冒号（全角或半角）作为“角度：”分隔
            positions = [p for p in (line.find("："), line.find(":")) if p != -1]
            sep_pos = min(positions) if positions else -1
            if sep_pos == -1:
                result.extend(wrap_text(line, width))
                continue
            prefix = line[: sep_pos + 1]
            content = line[sep_pos + 1 :].lstrip()
            if not content:
                result.append(prefix)
                continue
            chunks = wrap_text(content, width - display_width(prefix))
            result.append(prefix + chunks[0])
            for chunk in chunks[1:]:
                result.append(" " * HANG_INDENT + chunk)
        return "\n".join(result)

    origin = input("原提示词: ")
    res = from_natural_to_format(origin)
    print(f"格式化内容为：\n{format_with_hanging_indent(res)}")

