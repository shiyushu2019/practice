import os
import time
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")
dimensions=['文体','核心立意','事件与情节','主体与对象','时空与情境','陈述方式','结构节奏','语言风格','审美效果','创作语境']
MAX_RETRIES = 3

def from_natural_to_format(text):

    system_prompt = f"""你是一个语言格式化专家，善于把自然语言构成的创作指令转化为格式化要求。

一、分析角度（共{len(dimensions)}个，必须依次全部输出，一个都不能遗漏）：
{', '.join(dimensions)}

二、输出格式：
一共输出{len(dimensions)}行，每行格式为“角度名称：内容”。不要添加任何额外说明、标题或总结，不要在开头写“输出内容为：”之类，也不要在结尾写“以上就是格式化内容”之类，只依次输出这{len(dimensions)}行。

三、格式示例（仅示范格式，不代表内容）：
文体：散文
核心立意：表达极端孤独中的超然与坚守
事件与情节：记录了作者一次郊游，偶遇一只孤独的梅花，将自己的人生经历与梅花相类比
（示例中省略了后续行，但你必须{len(dimensions)}行全部输出）

四、填写规则：
1. 原提示词中明显含有的角度，必须如实、原样照搬，不得删减、想象或概括（除非原文实在过长）。例如原提示词说“我要写一个散文”，则“文体”行必须写“文体：散文”。
2. 原提示词未指明的字段，不得留空，必须以最不违和的方式自行创作补全。
3. 补全示例：若原提示词为“请帮我写一篇小说，内容是王小明今天上学迟到，被老师罚站的故事”，则“文体”写“小说”、“事件与情节”写“王小明今天上学迟到，被老师罚站的故事”、“主体与对象”写“王小明”；其余未明确字段（如核心立意）需自行合理创作补全，例如“核心立意：起到教育意义，启发孩子们不要迟到”。
4. 对未指明的字段尽量发散、写长，以达到能给创作者明确指示的程度，该类字段不得少于200字。
5. 所有字段的内容必须是确定的，不得出现“稍后确定”“根据实际情况展开”等偷懒说法。
6. 忽略原提示词中对篇幅的要求，不要因此改变格式化输出的篇幅。
7. 忽略原提示词中非创作要求的其他内容。"""

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
            if content is None:
                raise ValueError("API 返回空内容")
            return content
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError, OSError, ValueError) as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait_time = 2 ** (attempt - 1)
            print(f"[warning] 请求失败（第 {attempt} 次/共 {MAX_RETRIES} 次），{wait_time} 秒后重试：{e}")
            time.sleep(wait_time)

    raise RuntimeError(f"请求失败，已重试 {MAX_RETRIES} 次: {last_error}") if last_error else RuntimeError("请求失败")

if __name__ == "__main__":
    origin=input("原提示词: ")
    res=from_natural_to_format(origin)
    print(f"格式化内容为：\n{res}")

