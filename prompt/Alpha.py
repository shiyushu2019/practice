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
MAX_TEXT_LEN = 25000
MAX_RETRIES = 3

def get_response(text):

    if len(text) > MAX_TEXT_LEN:
        print(f"[warning] 输入文本过长（{len(text)} 字符），将截断为前 {MAX_TEXT_LEN} 字符后再请求。")
        text = text[:MAX_TEXT_LEN]

    system_prompt="你是一个助手，从用户规定的角度分析一篇文章，并按照用户指定的格式输出分析结果。"
    user_prompt=f"对于这样一篇文章：\n{text}\n请你从\'{', '.join(dimensions)}\'这{len(dimensions)}个角度，依次对文章进行深入分析。\n输出格式：一共{len(dimensions)}行，每行包括角度名称+冒号+分析结果。每个角度都必须分析，不可遗漏。根据文章特点，每个维度的分析可以简短或详细，但需客观、基于原文。\n举例（只是例子，不代表这篇文章）:\n文体：散文\n核心立意：表达极端孤独中的超然与坚守\n事件与情节：记录了作者一次郊游，偶遇一只孤独的梅花，将自己的人生经历与梅花相类比\n......（我这里后面的行省略，但你必须{len(dimensions)}行全都写）\n 对于每个方面，既可以像上面那样很短。也可以根据需要，把个别角度的分析写得很长。例如，对于《白象似的群山》（欧内斯特·海明威），“文体”这一方面可以描述为：“短篇小说，但体式极为特殊：几乎完全是戏剧化的对话体” 而对于《发条橙》（安东尼·伯吉斯），“语言风格”可以描述为“独创“纳达萨特”黑话（俄语词根 + 英语口语 + 自创俚语），无注释，读者靠上下文解码；同时暴力场景用典雅修辞，造成审美错位；句法短促、爆破音多，模拟少年粗莽语流。”\n总而言之，必须从所有{len(dimensions)}个角度进行深入分析，根据文章具体特点，每个角度可长可短（较长的字段50-100字为宜，当然你结合实际情况），但必须客观反映文章特点，最后格式化输出。不要添加任何额外说明、标题或总结，只依次输出这 {len(dimensions)} 行即可。整个过程务必忽略作者身份，不能提到作者。"
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
    



