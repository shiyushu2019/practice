import os
import time
import glob
import random
import json
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
from tqdm import tqdm

client = OpenAI(
        api_key=os.environ.get('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")
dimensions=['文体','核心立意','事件与情节','主体与对象','时空与情境','陈述方式','结构节奏','语言风格','审美效果','创作语境']
example_lines = "\n".join([f"{dim}：..." for dim in dimensions])
MAX_TEXT_LEN = 25000
MAX_RETRIES = 3

def get_response(text):

    system_prompt = f"""你是一个专业的文学分析助手。你的任务是严格按照下面规定的角度和格式，对用户提供的文章进行深入、客观的分析。

一、分析角度（共{len(dimensions)}个，必须依次全部分析，一个都不能遗漏）：
{', '.join(dimensions)}

二、分析规则：
1. 必须依次分析全部{len(dimensions)}个角度，不得遗漏任何一个。
2. 每个角度的分析必须客观、基于原文，反映文章特点。
3. 根据文章特点，每个角度可长可短：除“文体”字段只输出一个名词（如“散文”、“小说”，“诗歌”，等等）、不得有其他内容外，其他角度一般50-300字为宜，但你可根据实际情况自行权衡。
4. 忽略作者身份，不得提到作者；忽略出版情况，不得提到所获奖项、所在期刊等信息; 忽略与文章无关的其他信息；忽略署名、日期、后记、背景介绍，只看文章本身。
5. 严格遵循角度划分，例如，叙述视角只能写在“陈述方式”里，不要写在其它栏目，其他内容同理。

三、输出格式：
一共输出{len(dimensions)}行，每行格式为“角度名称：分析结果”。不要添加任何额外说明、标题或总结，只依次输出这{len(dimensions)}行。

四、输出格式示例（仅示范格式，这里用...代替内容，你必须基于输入文章独立分析）：
{example_lines}
示例中的...仅为占位符，表示该位置应填入具体分析文字，不代表实际输出长度或内容。
"""

    user_prompt = f"""请按照系统要求分析下面这篇文章：

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
            return content
        except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError, OSError, ValueError) as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break
            wait_time = 2 ** (attempt - 1)
            print(f"[warning] 请求失败（第 {attempt} 次/共 {MAX_RETRIES} 次），{wait_time} 秒后重试：{e}")
            time.sleep(wait_time)

    raise RuntimeError(f"请求失败，已重试 {MAX_RETRIES} 次: {last_error}") if last_error else RuntimeError("请求失败")
    

if __name__=="__main__":

    outputs=list()

    data_path="./data"
    file_list = sorted(glob.glob(f"{data_path}/**/*.txt", recursive=True))
    file_list=file_list[:10] # debug
    random.seed(123)
    random.shuffle(file_list)
    for fpath in tqdm(file_list):
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text=f.read()
                response=get_response(text)
                sample={
                    "description" : response,
                    "path": fpath,
                }
                outputs.append(sample)
        except Exception as e:
            print(f"error in {fpath}: {e}")
            continue
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    



