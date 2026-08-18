import os
import time
import glob
import random
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")
dimensions=['文体','核心立意','事件与情节','主体与对象','时空与情境','陈述方式','结构节奏','语言风格','审美效果','创作语境']
MAX_RETRIES = 3

def from_natural_to_format(text):

    system_prompt="你是一个语言格式化专家，善于把自然语言构成的指令转化为格式化要求。"
    user_prompt=f"""对于这样一个用户提交给语言模型的文学创作指令:

{text}

请你从\'{', '.join(dimensions)}\'这{len(dimensions)}个角度，对这个提示词进行概括。
输出格式：一共{len(dimensions)}行，每行包括一个字段，即角度名称+冒号+概括结果。举例（只是例子，不代表这个）:

文体：散文
核心立意：表达极端孤独中的超然与坚守
事件与情节：记录了作者一次郊游，偶遇一只孤独的梅花，将自己的人生经历与梅花相类比
......（我这里后面的行省略，但你必须{len(dimensions)}行全都写）

对于原提示词中明显含有的角度，你必须如实输出。例如，如果原提示词说“我要写一个散文，......”，你就必须在文体字段中这样写: 

文体：散文

对于明显含有的角度，你不能进行任何删减和想象，只能如实转述（注意不是概括，除非实在太长）

而如果用户的原提示词没有指明某个字段，你也不可以把这个字段（角度）空着，必须以最不违和的方法自己编一个。例如，如果用户原提示词的原话是“请帮我写一篇小说，内容是王小明今天上学迟到，被老师罚站的故事”，在这里很显然只明确敲定了文体（散文）、事件与情节（王小明今天上学迟到，被老师罚站的故事）、主体与对象（王小明）这几个字段，而其他字段都没有明确信息，这是你必须合理创作，将其他字段也写满，例如，核心立意部分可以这样写：“核心立意：起到教育意义，启发孩子们不要迟到，.....”其他部分同理。
总之，原则是基于原提示词、不删不改，要求是每个角度都必须写、不遗不漏，最终形成一个格式化的创作提示词。除了原提示词明显含有的角度你不能发散思考，其它的点（字段）你可以尽量写长一点，100字为宜，可以给创作者明确指示为宜。如果用户对篇幅有要求，请忽略，并不要因为用于原提示词对篇幅的要求而改变格式化输出的篇幅。
不要添加任何额外说明、标题或总结，不要在开头写“输出内容为：”之类，也不要在最后输出“以上就是格式化内容”之类，只依次输出这 {len(dimensions)} 行即可。忽略原提示词中非创作要求的其他内容。"""
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
                timeout=60,
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

