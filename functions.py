

import pandas as pd
import os
from dotenv import load_dotenv
#import dashscope
from typing import Dict, List, Callable
import re

import sys
from io import StringIO
import fitz  # PyMuPDF
import concurrent.futures
import copy

# 加载环境变量
load_dotenv()

from openai import OpenAI
print('导入依赖')
model = 'deepseek-r1-250528'
base_url = 'https://ark.cn-beijing.volces.com/api/v3'
api_key = os.getenv('HUOSHAN_API_KEY')







def get_resume(PDF):
    def extract_text_from_pdf(PDF):
 
        try:
            doc = fitz.open(PDF) # 使用 fitz.open() 打开PDF文件路径
            full_text = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # 使用 get_text("blocks") 获取文本块及其位置信息
                # 每个 block 是一个元组 (x0, y0, x1, y1, "text lines", block_no, block_type)
                # block_type: 0 for text, 1 for image
                blocks = page.get_text("blocks")
                page_lines = []
                for b in blocks:
                    if b[6] == 0: # 只处理文本块
                        # b[4] 包含该块内的所有文本行，每行以 '\n' 分隔
                        # 移除每行末尾的换行符，因为我们稍后会根据块来组织文本
                        lines_in_block = b[4].strip().split('\n')
                        page_lines.extend(lines_in_block)
                # 将页面中的所有行用换行符连接起来，形成一个完整的页面文本
                # 两个换行符用于分隔不同的页面
                full_text.append("\n".join(page_lines))
            return "\n\n".join(full_text)
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"
    
    extracted_text = extract_text_from_pdf(PDF)
    def fix_prompt_format(extracted_text) -> str:
        return f"""**你是AI简历助手应用中的一个函数，你需要执行以下任务，并按照格式输出结果**：
    
    # PDF解析格式修复

    ## 任务背景
    用户使用Python解析出来的PDF中的内容或多或少存在一定的格式混乱，需要您对其进行修复。该PDF的内容是一份简历，你需要根据自己的理解，整体把控该简历的内容，然后尝试还原该简历原本的格式。

    ## 任务要求
    1. 请根据您的理解，对该PDF的内容进行修复，修复解析混乱与错位的内容。
    2. 最终你需要给出修复好的简历，尽可能地使该简历还原为其原本的样子。
    3. 需要注意的是，除了修改错位的内容外，不要对原来的内容进行任何修改。

    ## 回复格式
    请在回复中，只包含修复好的简历内容，不要包含任何其他的内容。

    ## 待修复的PDF解析内容
    ```
    {extracted_text}
    ```
    """


    class fixformat:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,extracted_text) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = fix_prompt_format(extracted_text)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                # 添加替换逻辑：将 <br> 替换为 ●
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体




                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    fixformatAI = fixformat()

    while True:
        try:
            fixedformat = None
            fixedformat = fixformatAI.process_query(extracted_text)

            print(fixedformat)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")
    resume = fixedformat
    return resume


def get_daily_work(job):
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        # 从环境变量中获取您的 API Key
        api_key=os.environ.get("ARK_API_KEY")
    )
    research_theme_daily_work = f"""**你是AI简历助手应用中的一个函数，你需要执行以下任务，并按照格式输出结果**：
    
    # 任务背景
    你需要搜索并整理{job}岗位的日常工作内容，尽可能全面。

    # 任务要求
    1. 请搜索并整理{job}岗位的日常工作内容，尽可能全面。
    2. 在整理时，根据资料大致估计出这些日常性的工作分别占日常工作的百分之几。
    3. 请将您整理出来的日常工作内容，以markdown表格的形式呈现出来，包括“日常工作”、“详情”、“需要的技能”、“日常工作占比”四个字段，**其中“需要的技能”包括硬技能也包括软技能**。

    # 回复格式示例（需要严格遵守）
    ```
    | 日常工作 | 详情 | 需要的技能 | 日常工作占比 |
    | --- | --- | --- | --- |
    | 日常工作1 | 日常工作1的详情 | 技能1, 技能2 | 20% |
    | 日常工作2 | 日常工作2的详情 | 技能3, 技能4 | 30% |
    | 日常工作3 | 日常工作3的详情 | 技能5, 技能6 | 40% |
    | 其他杂活  | 日常工作3的详情 | 技能5, 技能6 | 10% |

    其他补充与说明（如有）：...
    ```
    **表格内容以叙述形式呈现，禁止在表格内使用<li>、<ul>、<ol>、<br>等标签与语法。**
    """
    stream = client.chat.completions.create(
        model="bot-20250605153443-w9vcg",  # bot-20250605153443-w9vcg 为您当前的智能体的ID，注意此处与Chat API存在差异。差异对比详见 SDK使用指南
        messages=[
            {"role": "user", "content": research_theme_daily_work},
        ],
        stream=False,
    )
    if stream.choices and stream.choices[0].message and stream.choices[0].message.content:
        daily_work = stream.choices[0].message.content
        print(daily_work) # 保持打印行为，如果需要的话
        # 如果需要处理 references，可以检查 stream.references
        if hasattr(stream, "references"):
            print(stream.references)
    else:
        daily_work = ""
    return daily_work

def get_interview(job):
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        # 从环境变量中获取您的 API Key
        api_key=os.environ.get("ARK_API_KEY")
    )
    research_theme_interview = f"""**你是AI简历助手应用中的一个函数，你需要执行以下任务，并按照格式输出结果**：
    
    # 任务背景
    你需要搜索并整理近期{job}岗位的笔面试题、笔面试考察技能、通过员工的画像等，尽可能全面。

    # 任务要求
    1. 请搜索近期{job}岗位的笔面试题、笔面试考察技能、通过员工的画像等，尽可能全面。
    2. 根据资料推测出为了迎接面试笔试，需要学习/温习/突击哪些知识，分别需要熟悉到哪种程度。
    3. 在回复中，你需要整理出笔面试大致考察的知识与技能的范围，以markdown表格的形式呈现出来，包括“需要准备的知识与技能”、“详情”、“需要熟悉到何种程度”、“笔面试考察占比”四个字段，**其中“需要的技能”包括硬技能也包括软技能**。

    # 回复格式（需要严格遵循，不要输出其他内容）
    ```
    | 需要准备的知识与技能 | 详情 | 需要熟悉到何种程度 | 笔面试考察占比 |
    | --- | --- | --- | --- |
    | 知识1 | 知识1的详情 | 熟悉程度1 | 20% |
    | 知识2 | 知识2的详情 | 熟悉程度2 | 30% |
    | 知识3 | 知识3的详情 | 熟悉程度3 | 40% |

    其他补充与说明（如有）：...
    ```
    **表格内容以叙述形式呈现，禁止在表格内使用<li>、<ul>、<ol>、<br>等标签与语法。**
    """
    stream = client.chat.completions.create(
        model="bot-20250605153443-w9vcg",  # bot-20250605153443-w9vcg 为您当前的智能体的ID，注意此处与Chat API存在差异。差异对比详见 SDK使用指南
        messages=[
            {"role": "user", "content": research_theme_interview},
        ],
        stream=False,
    )
    if stream.choices and stream.choices[0].message and stream.choices[0].message.content:
        interview = stream.choices[0].message.content
        print(interview) # 保持打印行为
        if hasattr(stream, "references"):
            print(stream.references)
    else:
        interview = ""
    return interview


def get_peer_resume(job):
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        # 从环境变量中获取您的 API Key
        api_key=os.environ.get("ARK_API_KEY")
    )
    research_theme_peer_resume = f"""**你是AI简历助手应用中的一个函数，你需要执行以下任务，并按照格式输出结果**：
    
    # 任务背景
    你需要搜索并整理近年来前去应聘{job}岗位的求职者的简历画像，他们通常都有着什么样的经历、身份与技能，尽可能全面。

    # 任务要求
    1. 请搜索并整理近年来前去应聘{job}岗位的求职者通常都有着什么样的经历、身份与技能，他们的简历大概都是怎样的。
    2. 你需要注意大多数求职者的简历的画像，以及其中优秀的求职者的简历画像。
    3. 请将你整理出来的各层次的简历的画像，以markdown表格的形式呈现出来，包括“求职者分类”、“简历画像概述”、“简历画像详情”三个字段，**其中“求职者分类”包括“大多数求职者”与“优秀求职者”，简历画像详情中需要突出优秀求职者与大多数求职者的区别**。

    # 回复格式示例（需要严格遵守）
    
    ```
    | 求职者分类 | 简历画像概述 | 简历画像详情 |
    | --- | --- | --- |
    | 大多数求职者 | 大多数求职者的简历画像概述 | 大多数求职者的简历画像详情 |
    | 优秀求职者 | 优秀求职者的简历画像概述 | 优秀求职者的简历画像详情 |

    其他补充与说明（如有）：...
    ```
    **表格内容以叙述形式呈现，禁止在表格内使用<li>、<ul>、<ol>、<br>等标签与语法。**
    """
    stream = client.chat.completions.create(
        model="bot-20250605153443-w9vcg",  # bot-20250605153443-w9vcg 为您当前的智能体的ID，注意此处与Chat API存在差异。差异对比详见 SDK使用指南
        messages=[
            {"role": "user", "content": research_theme_peer_resume},
        ],
        stream=False,
    )
    if stream.choices and stream.choices[0].message and stream.choices[0].message.content:
        peer_resume = stream.choices[0].message.content
        print(peer_resume) # 保持打印行为
        if hasattr(stream, "references"):
            print(stream.references)
    else:
        peer_resume = ""
    return peer_resume


def get_resume_match(job):
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3/bots",
        # 从环境变量中获取您的 API Key
        api_key=os.environ.get("ARK_API_KEY")
    )
    research_theme_resume_match = f"""**你是AI简历助手应用中的一个函数，你需要执行以下任务，并按照格式输出结果**：

    # 任务背景
    你需要搜索并整理应聘{job}岗位的简历应该怎么写，包括哪些内容，每一部分内容需要突出什么特质，尽可能全面。

    # 任务要求
    1. 请搜索并整理应聘{job}岗位的简历应该怎么写，包括哪些内容，每一部分内容需要突出什么特质，尽可能全面。
    2. 你需要甄别所搜集的资料是否都是合理的，有可能有的资料中所说的{job}岗位的简历模版并不符合实际简历规范，并不有助于写出一份优秀的{job}简历
    3. 请将你整理出来的简历需要哪些部分、各部分需要怎么写，以markdown表格的形式呈现出来，包括“简历所需模块”、“该模块需要怎么写”两个字段。

    # 回复格式示例（需要严格遵守）

    ```
    | 简历所需模块 | 该模块需要怎么写 |
    | --- | --- |
    | 模块1（如学历） | 模块1的内容（如需要突出在校奖项） |
    | 模块2 | 模块2的内容 |
    | 模块3 | 模块3的内容 |

    其他补充与说明（如有）：...
    ```
    **表格内容以叙述形式呈现，禁止在表格内使用<li>、<ul>、<ol>、<br>等标签与语法。**
    """
    stream = client.chat.completions.create(
        model="bot-20250605153443-w9vcg",  # bot-20250605153443-w9vcg 为您当前的智能体的ID，注意此处与Chat API存在差异。差异对比详见 SDK使用指南
        messages=[
            {"role": "user", "content": research_theme_resume_match},
        ],
        stream=False,
    )
    if stream.choices and stream.choices[0].message and stream.choices[0].message.content:
        resume_match = stream.choices[0].message.content
        print(resume_match) # 保持打印行为
        if hasattr(stream, "references"):
            print(stream.references)
    else:
        resume_match = ""
    return resume_match

def run_all_functions_parallel(job):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务到线程池
        future_daily_work = executor.submit(get_daily_work, job)
        future_interview = executor.submit(get_interview, job)
        future_peer_resume = executor.submit(get_peer_resume, job)
        future_resume_match = executor.submit(get_resume_match, job)

        # 获取结果
        results['daily_work'] = future_daily_work.result()
        results['interview'] = future_interview.result()
        results['peer_resume'] = future_peer_resume.result()
        results['resume_match'] = future_resume_match.result()

    return results


def get_daily_work_rate(job, resume, results):
    daily_work = results['daily_work']
    def daily_work_rate_prompt_format(job, resume, daily_work) -> str:
        return f"""
    # 日常工作胜任力评分任务

    ## 任务背景
    你需要根据{job}岗位的日常工作内容，以及用户的简历内容，来评估用户对{job}岗位的日常工作胜任力（百分制）。

    ## 任务要求
    1. 请根据{job}岗位的日常工作内容，以及用户的简历内容，来评估用户对{job}岗位的日常工作胜任力（百分制）。
    2. 不要太局限于用户的经历是否对口，你需要关注简历中体现出的用户的能力与技能，看其是否能胜任完成日常工作，或者说能胜任哪些日常工作，而哪些方面还有点欠缺。
    3. 最终给出一个百分制的分数，分数越高，说明简历中体现出用户对{job}岗位的日常工作胜任力越好，60分及格，60-70分表示勉强能完成一些基础工作，70-80分表示能胜任大多数的日常工作，80-90分表示能高效地完成绝大多数日常工作，90分以上表示能应对一些更深层次、更困难的工作需求。
    4. 在给出分数的同时，你需要给出一份评分报告，有条理地叙述你为何给出这样的分数，包括哪些日常工作是胜任的，哪些日常工作是无法较好地完成等等。

    ## 用户简历与{job}日常工作内容
    - 用户的简历内容如下：
    ```
    {resume}
    ```
    - {job}岗位的日常工作内容如下：
    ```
    {daily_work}
    ```

    ## 回复格式
    请在回复中，只包含分数与评分报告，不要包含任何其他的内容。回复示例如下：
    ```
    日常工作胜任力评分：80
    评分报告：
    用户整体上能够较好地胜任···（概况）
    用户在简历中所体现出的能力与技能，能够胜任...等工作，而对于更深层次的...等工作，用户可能还需进一步学习。（详细分析报告）
    ```
    以上回复示例仅做简单说明，你需要具体内容具体分析，给出更详实的评分报告。
    """


    class daily_work_rate:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,job, resume, daily_work) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = daily_work_rate_prompt_format(job, resume, daily_work)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体




                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    daily_work_rate_AI = daily_work_rate()

    while True:
        try:
            daily_work_rate = None
            daily_work_rate = daily_work_rate_AI.process_query(job, resume, daily_work)

            print(daily_work_rate)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")
    return daily_work_rate


def get_interview_pass_rate(job, resume, results):
    interview = results['interview']
    def interview_pass_rate_prompt_format(job, resume, interview) -> str:
        return f"""
    # 笔面试通过率评分任务

    ## 任务背景
    你需要根据{job}岗位的笔面试考察内容，以及用户的简历内容，来评估用户对{job}岗位的笔面试通过率。

    ## 任务要求
    1. 请根据{job}岗位的笔面试考察内容，以及用户的简历内容，来评估用户对{job}岗位的笔面试通过率。
    2. 你需要关注简历中体现出用户可能对面试所需的哪些方面的知识与技能熟悉或精通，那么这部分的知识与技能用户准备起来应该会相对容易；用户对所需的哪些方面的知识与技能相对欠缺，若这些知识与技能在面试与笔试中又比较重要，那么用户需要着重学习与准备这方面的知识与技能。
    3. 需要注意的是，有些职业知识与技能非常具体、琐碎、繁杂，那么这些知识与技能通常非常容易遗忘，即便用户过往经历体现出其掌握这样的知识与技能，在找新的工作时，依旧需要注意温习这些“八股文”。
    4. 最终给出一个百分制的分数，分数越高，说明简历中体现出用户越容易通过{job}岗位的笔面试，60分及格，60-70分表示有少部分的知识与技能用户可能没有接触过，需要补足，70-80分表示用户基本接触过笔面试所需知识与技能，但接触程度较浅，需要扎实的准备，80-90分表示用户经常接触笔面试所需的知识与技能，仅需花费较少时间准备笔面试，90分以上表示用户的过往经历出彩，甚至面试官会觉得不必太严格考察用户的基础知识与技能了。
    5. 在给出分数的同时，你需要给出一份评分报告，有条理地叙述你为何给出这样的分数，包括用户对所需的哪些知识与技能熟悉、哪些需要补足，用户需要着重准备哪些知识与技能等等。

    ## 用户简历与{job}岗位笔面试内容
    - 用户的简历内容如下：
    ```
    {resume}
    ```
    - {job}岗位的笔面试内容如下：
    ```
    {interview}
    ```

    ## 回复格式
    请在回复中，只包含分数与评分报告，不要包含任何其他的内容。回复示例如下：
    ```
    笔面试通过率评分：80
    评分报告：
    用户在简历中体现出了较强的能力···（概况）
    用户可能已经较为熟悉...等知识与技能，需要花部分时间去温习这些知识，因为目标岗位对这些知识与技能的考察较为固定与重视（面试八股文）。而对于...等知识与技能，用户可能还不熟悉，需花较多时间去学习与准备。（详细分析报告）
    ```
    以上回复示例仅做简单说明，你需要具体内容具体分析，给出更详实的评分报告。
    """


    class interview_pass_rate:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,job, resume, interview) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = interview_pass_rate_prompt_format(job, resume, interview)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体




                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    interview_pass_rate_AI = interview_pass_rate()

    while True:
        try:
            interview_pass_rate = None
            interview_pass_rate = interview_pass_rate_AI.process_query(job, resume, interview)

            print(interview_pass_rate)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")
    return interview_pass_rate


def get_peer_pressure_rate(job, resume, results):
    peer_resume = results['peer_resume']
    def peer_pressure_rate_prompt_format(job, resume, peer_resume) -> str:
        return f"""
    # 同侪竞争出众度评分任务

    ## 任务背景
    你需要根据前往应聘{job}岗位的求职者的简历画像，以及用户的简历内容，来评估用户应聘{job}岗位时的同侪竞争出众度（百分制）。

    ## 任务要求
    1. 请根据前往应聘{job}岗位的求职者的简历画像，以及用户的简历内容，来评估用户应聘{job}岗位时的同侪竞争出众度（百分制）。
    2. 请站在面试官的角度，结合该岗位所需求职者的特征，思考与对比：用户的简历内容相比于大多数求职者是否更加契合该岗位、相比于较优秀求职者是否更加契合该岗位。
    3. 最终给出一个百分制的分数，分数越高，说明简历中体现出用户在同侪中对{job}岗位的竞争力越强，60分及格，60-70分表示勉强跟得上大多数求职者的简历水平，70-80分表示用户的简历超出了大多数求职者的简历，不过可能稍逊于优秀求职者的水平，80-90分表示用户的简历完全能称得上是优秀的求职者，90分以上表示用户的简历相比于一般的优秀求职者更加出众。
    4. 在给出分数的同时，你需要给出一份评分报告，有条理地叙述你为何给出这样的分数，包括用户的简历在应聘{job}的同侪中处于什么水平，用户的简历可能还欠缺哪方面等等。

    ## 用户简历与{job}同侪简历画像
    - 用户的简历内容如下：
    ```
    {resume}
    ```
    - {job}岗位的同侪简历画像如下：
    ```
    {peer_resume}
    ```

    ## 回复格式
    请在回复中，只包含分数与评分报告，不要包含任何其他的内容。回复示例如下：
    ```
    同侪竞争出众度评分：75
    评分报告：
    用户的简历在同侪中处于中等水平···（概况）
    相较于大多数求职者的简历，用户的简历有...等优势，但可能还欠缺一些...等方面，而相较于优秀求职者的简历，用户的简历可能还欠缺一些...等方面。（详细分析报告）
    ```
    以上回复示例仅做简单说明，你需要具体内容具体分析，给出更详实的评分报告。
    """


    class peer_pressure_rate:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,job, resume, peer_resume) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = peer_pressure_rate_prompt_format(job, resume, peer_resume)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体




                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    peer_pressure_rate_AI = peer_pressure_rate()

    while True:
        try:
            peer_pressure_rate = None
            peer_pressure_rate = peer_pressure_rate_AI.process_query(job, resume, peer_resume)

            print(peer_pressure_rate)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")
    return peer_pressure_rate


def get_resume_match_rate(job, resume, results):
    resume_match = results['resume_match']

    def resume_match_rate_prompt_format(job, resume, resume_match) -> str:
        return f"""
    # 简历内容与职位匹配度评分任务

    ## 任务背景
    你需要根据{job}岗位的所偏好的简历模板与特征，以及用户的简历内容，来评估用户的简历内容有多契合{job}岗位的偏好（百分制）。

    ## 任务要求
    1. 请根据{job}岗位的所偏好的简历模板与特征，以及用户的简历内容，来评估用户的简历内容有多契合{job}岗位的偏好（百分制）。
    2. 最终给出一个百分制的分数，分数越高，说明用户的简历越契合{job}岗位的偏好，60分及格，60-70分表示勉强契合{job}所偏好的简历模板与特征，70-80分表示用户的简历基本符合{job}所偏好的简历模板与特征，不过在描述上可能有所欠缺，80-90分表示用户的简历在内容上、结构上、描述上都相当契合{job}所偏好的简历模板与特征，90分以上表示用户的简历在更深的细节方面也做得很有亮点，能够很大程度地吸引面试官。
    3. 在给出分数的同时，你需要给出一份评分报告，有条理地叙述你为何给出这样的分数，包括用户的简历在哪些方面做得可圈可点，在哪些方面做得欠缺，需要改进等等。

    ## 用户简历与{job}偏好的简历模板与特征
    - 用户的简历内容如下：
    ```
    {resume}
    ```
    - {job}岗位偏好的简历模板与特征如下：
    ```
    {resume_match}
    ```

    ## 回复格式
    请在回复中，只包含分数与评分报告，不要包含任何其他的内容。回复示例如下：
    ```
    简历内容与职位匹配度评分：75
    评分报告：
    用户的简历在多方面匹配目标岗位，但在···（概况）
    用户的简历在内容方面做得比较全面，在教育背景、项目经验、获奖情况等方面都有体现，不过在技能方面可能还欠缺一些，其描述没有很好地展现其技能具体到了什么样的程度。（详细分析报告）
    ```
    以上回复示例仅做简单说明，你需要具体内容具体分析，给出更详实的评分报告。
    """


    class resume_match_rate:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,job, resume, resume_match) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = resume_match_rate_prompt_format(job, resume, resume_match)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体




                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    resume_match_rate_AI = resume_match_rate()

    while True:
        try:
            resume_match_rate = None
            resume_match_rate = resume_match_rate_AI.process_query(job, resume, resume_match)

            print(resume_match_rate)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")
    return resume_match_rate

def run_all_rate_functions_parallel(job, resume, results):
    rate_results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务到线程池
        future_daily_work_rate = executor.submit(get_daily_work_rate, job, resume, results)
        future_interview_pass_rate = executor.submit(get_interview_pass_rate, job, resume, results)
        future_peer_pressure_rate = executor.submit(get_peer_pressure_rate, job, resume, results)
        future_resume_match_rate = executor.submit(get_resume_match_rate, job, resume, results)

        # 获取结果
        rate_results['daily_work_rate'] = future_daily_work_rate.result()
        rate_results['interview_pass_rate'] = future_interview_pass_rate.result()
        rate_results['peer_pressure_rate'] = future_peer_pressure_rate.result()
        rate_results['resume_match_rate'] = future_resume_match_rate.result()

    return rate_results

def get_resume_and_initial_results_parallel(PDF, job):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_resume = executor.submit(get_resume, PDF)
        future_initial_results = executor.submit(run_all_functions_parallel, job)

        resume_text = future_resume.result()
        initial_results = future_initial_results.result()
        
    return resume_text, initial_results








def polish_resume(job_title,resume_text,initial_results,all_rate_results,personalization):

    """
    让AI给出包装建议
    """

    def polish_resume_prompt_format(job_title,resume_text,initial_results,all_rate_results,personalization) -> str:
        
        daily_work = initial_results['daily_work'] + "\n" + all_rate_results['daily_work_rate']
        interview_pass = initial_results['interview'] + "\n" + all_rate_results['interview_pass_rate']
        peer_pressure = initial_results['peer_resume'] + "\n" + all_rate_results['peer_pressure_rate']
        resume_match = initial_results['resume_match'] + "\n" + all_rate_results['resume_match_rate']
        personalization_c = copy.deepcopy(personalization["包装程度"])
        if personalization["包装程度"] == "高风险":
            personalization_c = "**高风险包装策略，在用户原经历的基础上，进行适当的延展与虚构，使用户的经历更符合岗位的要求。**"
        elif personalization["包装程度"] == "适度包装":
            personalization_c = "**适度包装策略，保持原经历的核心内容并进行一定的修饰、改写与迁移，不过度包装，可以适当虚构边缘的经历，但不能编造核心经历、核心经过，最终使内容更契合岗位的同时难以被质疑。**"
        else:
            personalization_c = "**真实模式，真实地保持原有内容而几乎不进行虚构与夸张，重点放在优化语言表达，使内容更贴近岗位。**"

        print(personalization)
        print(personalization_c)

        return f"""你是AI简历助手应用中的一个函数，你需要遵循以下函数定义，接收输入、执行任务并输出结果：

    ---

    # 函数定义

    ## 任务背景

    请根据用户的简历内容、用户的简历评估结果、用户的包装偏好，给出一份包装建议，为后续用户的简历修改指明方向。
    
    ## 任务理解

    所谓“包装”简历，就是对简历中的经历描述进行多维度的优化，使其更加贴合目标岗位的要求。这样多维度的优化也正是本次任务的核心价值点所在，你可以从以下维度把握“包装”的内涵：
     1. 用户可能并没有目标岗位的经历，但用户的某些经历与目标岗位的工作在内核上其实有相近之处，此时可以将用户的真实经历改头换面（例如修改岗位或项目名称，重构项目情景、目的、结果，优化工作内容描述等等），包装用户的真实经历，使其尽量往目标岗位靠拢。
     2. 用户可能有目标岗位的相关经历，但用户的经历或许不够出色，此时可以对用户的经历描述进行优化与提升（例如将成果进行适当的夸张，或增添部分细节，或将普通的工作内容描述得具有挑战性、更专业等等），需注意描述脱离实际会提高被质疑的风险（例如像学历、证书、奖章等就不能进行夸张）。
    在包装简历的实操过程中需要注意凸显项目的成果的价值。而如何凸显项目成果的价值，其中也有门道，关键在于成果的“可验证性”：
     1. 若用户项目成果的价值已有外部数据支持，可直接使用外部数据支持，HR可以通过这些显性的、可验证的数据来直观感受项目成果的价值。
     2. **若用户的项目成果没有出众的外部数据展现，想要通过包装既显得项目成果丰硕，又不易被外部验证，那么核心在于：用模糊而积极的量化词汇，结合定性描述，突出内部价值和复杂性，而非直接可查的外部数据，同时也避免过分夸大。**
    但是，包装简历其实并不容易，非常容易包装出脱离实际、用户难以接受的经历，例如将经历包装得不符合存在逻辑、完全脱离用户的接触范围，导致用户完全不理解、不了解包装后的经历到底是个什么样的事件。
    那么，包装时要怎么做到包装出的经历是**符合存在逻辑**、**符合岗位需求**又**让用户易于接受**的呢？这需要遵循以下的思考步骤：
     1. 第一步：提炼真实经历的核心过程与核心能力 首先需要深入挖掘用户真实工作经历中的核心过程和核心能力，这些是用户实打实了解和掌握的部分。这一步的关键在于识别出用户在项目中真正参与的技术实现过程、解决问题的思路方法、使用的具体工具和技术栈，以及在团队协作中发挥的实际作用。包装后的经历需要大部分地保留这些核心过程（但不必保留原经历的背景、任务、结果等），这样用户在面试时才能有底气、有逻辑，不怕面试官的细节追问，因为这些都是基于真实经验的。
     2. 第二步：分析目标岗位偏好与经历重合度并提出包装的可选大方向 深入思考目标岗位究竟偏好什么样的项目经历和能力背景，然后仔细分析用户当前真实经历的核心过程是否与这些偏好有重合点或者相关性。即使看起来不太相关的经历，也要挖掘其中可能与目标岗位沾边的技术点、业务逻辑或者能力要求。这个分析过程将为用户的简历包装提供几个可选的发展方向，每个方向可能基于真实经历的某些侧面。
     3. 第三步：研究真实经历的公司背景与业务逻辑并筛选出符合逻辑的包装方向 思考用户真实工作过的公司究竟是做什么业务的，可能涉及哪些具体的业务场景和技术需求，公司的规模、发展阶段、技术架构选择等背景信息。这些背景信息将为包装后的项目经历提供坚实的基础支撑，确保编造的项目在该公司的实际背景下是符合存在逻辑的，不会让人一眼就看出破绽或感到不合理。
     4. 第四步：进一步思考并筛选出最优的、最合适的包装方向进行细化打磨 从前面分析得出的几个可选包装方向中，选择最恰当、最容易包装得真实可信的那个方向。要综合考虑包装的难度、风险和效果，选择既能有效提升与目标岗位的匹配度，又不会过度偏离真实背景的方向。然后具体设计原有项目经历应该如何进行包装，包括项目背景的重新设定、技术难点的重新包装、个人贡献的合理放大等。
    同时也需要遵循以下原则：
     - 运用编故事思维进行移花接木 包装简历经历本质上需要有编故事的思维能力，核心技巧是将用户原有经历中的真实核心过程移花接木到新编造的项目故事中。这个新故事可以为用户的经历赋予完全不同的业务背景、项目目标、应用场景，但必须保持技术实现的核心逻辑不变。同时这个故事必须遵循基本的商业逻辑和技术逻辑，要符合公司的整体背景、业务发展需要、技术架构选择等大的框架约束，确保故事的可信度和合理性。
     - 真实内核不变原则 无论如何包装，都必须保持技术实现的核心逻辑和解决问题的基本思路不变。可以改变项目的业务背景、应用场景、团队规模，但不能改变你实际掌握的技术能力和解决问题的方法。这是包装可信度的根本保障，也是面试时能够深入回答技术问题的基础。
     - 适度放大原则 包装时要把握好度，避免过度夸大。一般来说，影响范围可以适度扩大，技术难度可以合理提升，个人贡献可以突出强调，但都要在合理的范围内。比如一个5人团队的项目不要包装成50人的大项目，一个内部工具不要包装成面向百万用户的系统。
     - 逻辑自洽原则 包装后的所有内容必须在逻辑上能够自圆其说。项目的时间周期要与复杂度匹配，个人的角色要与经验水平匹配，技术选型要与公司背景匹配，团队规模要与项目规模匹配。任何一个环节出现逻辑矛盾，都可能让整个包装失去可信度。

    ## 任务要求

    - 用户的简历与岗位或多或少存在着不匹配，用户已经做了初步的简历诊断，你需要根据用户的简历诊断报告，以及用户的包装偏好，为用户指出其简历中的各个项目可以往什么方向去包装与优化，你只需指明大致方向与要点，而不必具体修改。
    - **对于用户的每一个占据一定介绍篇幅的项目经历，都需要进行包装指导，在一家公司中的不同项目经历也需要分别进行包装指导，而不能盲目地将在一家公司的经历不加思考地视为一个整体。**
    - **尤其需要注意的是，你给出包装建议包括两个方面，“包装方向”与“思路”，“包装方向”中需要说明该项目可以往什么方向进行包装，“思路”中可以阐述包装时需要注意些什么（该如何去包装）、原项目经历有哪些特点所以才适合向该方向进行包装等等。若某些经历实在与目标岗位不匹配，也无法体现用户的突出能力，可以告知用户该经历可以删除。**
    - **允许提出“嫁接与合并”的包装建议，将用户的质量不高的经历嫁接到另一段经历中，并进行整合包装，从而得到一个质量更高的经历，这样的经历通常比两个低质量的经历更有效。**
    - **对于质量较低的经历，首先考虑其能否通过嫁接与合并的方式进行包装，若两个经历实在难以相融，才考虑单独进行包装或者直接删除。**
    - 为了确保给的建议能够指导实践，请在包装经历时，对每一个经历，都作出如下的思考：1. 提炼真实经历的核心过程与核心能力；2. 分析目标岗位偏好与经历重合度并提出包装的可选大方向；3. 研究真实经历的公司背景与业务逻辑并筛选出符合逻辑的包装方向；4. 进一步思考并筛选出最优的、最合适的包装方向进行细化打磨。
    - 为了确保给的建议能够指导实践，请在包装经历时，遵循以下原则：真实内核不变原则、适度放大原则、逻辑自洽原则。
    - 牢记最重要的原则——运用编故事思维进行移花接木：对每一个经历的包装其实都是编一个半真半假的新故事，该故事核心经过、核心能力为真，但其他部分可作修饰或编造。

    ## 输出格式与示例

    你的输出格式为一个md表格，该表格有三列：“序号”、“原项目经历”、“修改建议”。“原项目经历”中**若提出嫁接与合并的建议，则同时填写两个嫁接合并的项目**。“修改建议”中则是针对该项目经历的包装与修改建议，你需要说明：说明该项目可以往什么方向进行包装，包装时需要注意该如何去包装、用户原经历为什么适合这样包装等等。若某些经历实在与目标岗位不匹配，也无法体现用户的突出能力，还难以与其他项目经历进行嫁接，可以告知用户该经历可以删除。
    **需谨记，你是AI简历助手应用中的一个函数，后续程序会检测你输出中的md表格，表格外的一切内容都会被忽略，因此你需要将内容都放在该md表格中，无需在表格外叙述其他内容，也决不允许输出额外的md表格。**
    输出示例（尤其注意嫁接合并的写法）：

    |序号|原项目经历|修改建议|
    |---|---|---|
    |1|项目经历1|修改建议1|
    |2|项目经历2|修改建议2|
    |3|项目经历3|修改建议3|
    |4|低质经历a+经历b|将a嫁接与整合进b中的建议4|

    **表格内容以叙述形式呈现，禁止在表格内使用<li>、<ul>、<ol>、<br>等标签与语法。**

    ---

    # 输入

    ## 用户目标岗位

{job_title}

    ## 用户简历

{resume_text}

    ## 简历诊断报告（请重点参考）

    ```日常工作胜任力评估报告
{daily_work}
    ```

    ```笔面门槛通过率评估报告
{interview_pass}
    ```

    ```同侪竞争出众度评估报告
{peer_pressure}
    ```

    ```简历内容匹配度评估报告
{resume_match}
    ```

    ## 包装偏好
    
{personalization_c}
**请重点参考简历诊断报告，在包装与修改简历的时候，尽量往简历诊断报告中提到的技能、经验、能力上面靠拢。**
**请先检查是否有低质量的项目（低能力体现、低难度、低相关度的项目即是低质量项目），若有低质量项目请首先考虑是否能嫁接融合到其他项目中，尝试一起包装为更优秀的项目，若实在冲突，则可建议直接删除。**
    ---
    
    **执行以上任务。**
    """


    class polish_resume:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,job_title,resume_text,initial_results,all_rate_results,personalization) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = polish_resume_prompt_format(job_title,resume_text,initial_results,all_rate_results,personalization)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                # 添加替换逻辑：将 <br> 替换为 ●
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体



                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    polish_resume_AI = polish_resume()

    while True:
        try:
            polished_resume = None
            polished_resume = polish_resume_AI.process_query(job_title,resume_text,initial_results,all_rate_results,personalization)

            print(polished_resume)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")


    # 提取markdown表格部分
    lines = polished_resume.split('\n')
    table_lines = []
    in_table = False

    for line in lines:
        line = line.strip()
        
        # 表格开始标志
        if '|序号|' in line and '|' in line:
            in_table = True
            table_lines.append(line)
        # 在表格中且是表格行
        elif in_table and '|' in line and line:
            table_lines.append(line)
        # 在表格中但是空行，跳过
        elif in_table and not line:
            continue
        # 在表格中但遇到非表格行，结束
        elif in_table:
            break

    polished_resume = '\n'.join(table_lines) if table_lines else ""
        
    return polished_resume



def update_polish_suggestions(user_message, current_marker,original_polish_suggetions,job_title,resume_text,initial_results,all_rate_results,personalization):
    """基于用户的反馈，AI修改计划。"""

    def extract_project_experience_by_marker(original_polish_suggetions, current_marker):
        """
        从markdown表格字符串中提取指定序号对应的项目经历
        
        Args:
            original_polish_suggetions (str): markdown表格字符串
            current_marker (str): 序号字符串
        
        Returns:
            str: 对应序号的项目经历，如果未找到则返回None
        """
        try:
            # 将current_marker转换为整数
            target_index = int(current_marker)
            
            # 按行分割表格内容
            lines = original_polish_suggetions.strip().split('\n')
            
            # 找到表格数据行（跳过表头和分隔符行）
            data_lines = []
            for line in lines:
                line = line.strip()
                # 跳过空行、表头行和分隔符行
                if line and not line.startswith('|序号|') and not line.startswith('|---|'):
                    data_lines.append(line)
            
            # 遍历数据行，查找匹配的序号
            for line in data_lines:
                if line.startswith('|') and line.endswith('|'):
                    # 分割表格列
                    columns = [col.strip() for col in line.split('|')[1:-1]]  # 去掉首尾空元素
                    
                    if len(columns) >= 2:  # 确保至少有序号和项目经历两列
                        try:
                            # 检查序号是否匹配
                            if int(columns[0]) == target_index:
                                return columns[1]  # 返回项目经历列
                        except ValueError:
                            continue  # 如果序号不是数字，跳过这行
            
            return None  # 未找到匹配的序号
            
        except ValueError:
            print(f"错误：current_marker '{current_marker}' 不是有效的数字")
            return None
        except Exception as e:
            print(f"提取项目经历时发生错误：{str(e)}")
            return None
    
    def update_polish_resume_prompt_format(user_message, current_marker,original_polish_suggetions,job_title,resume_text,initial_results,all_rate_results,personalization) -> str:
        

        
        daily_work = initial_results['daily_work'] + "\n" + all_rate_results['daily_work_rate']
        interview_pass = initial_results['interview'] + "\n" + all_rate_results['interview_pass_rate']
        peer_pressure = initial_results['peer_resume'] + "\n" + all_rate_results['peer_pressure_rate']
        resume_match = initial_results['resume_match'] + "\n" + all_rate_results['resume_match_rate']
        personalization_c = copy.deepcopy(personalization["包装程度"])
        if personalization["包装程度"] == "高风险":
            personalization_c = "**高风险包装策略，在用户原经历的基础上，进行适当的延展与虚构，使用户的经历更符合岗位的要求。**"
        elif personalization["包装程度"] == "适度包装":
            personalization_c = "**适度包装策略，保持原经历的核心内容并进行一定的修饰、改写与迁移，不过度包装，可以适当虚构边缘的经历（虚构数据时一定要谨慎，虚构的数据一定要匹配用户的能力水平），但不能编造核心经历、核心经过，最终使内容更契合岗位的同时难以被质疑。**"
        else:
            personalization_c = "**真实模式，真实地保持原有内容而几乎不进行虚构与夸张，重点放在优化语言表达，使内容更贴近岗位。**"

        current_project_experience = extract_project_experience_by_marker(original_polish_suggetions, current_marker)


        print(personalization)
        print(personalization_c)
        return f"""你是AI简历助手应用中的一个函数，你需要遵循以下函数定义，接收输入、执行任务并输出结果：

    ---

    # 函数定义

    ## 任务背景

    在AI简历助手应用的前一个环节，我们已经向用户提供了一份简历修改方案，用户经浏览后，停留在其中一个页面上，向我们提出了反馈（建议或不满等），请根据用户简历相关的背景资料、之前提供给用户的简历修改方案、用户所在页面位置，结合用户的反馈和用户包装简历的偏好，调整之前的简历修改方案。
    
    ## 任务理解

    所谓“包装”简历，就是对简历中的经历描述进行多维度的优化，使其更加贴合目标岗位的要求。这样多维度的优化也正是本次任务的核心价值点所在，你可以从以下维度把握“包装”的内涵：
     1. 用户可能并没有目标岗位的经历，但用户的某些经历与目标岗位的工作在内核上其实有相近之处，此时可以将用户的真实经历改头换面（例如修改岗位或项目名称，重构项目情景、目的、结果，优化工作内容描述等等），包装用户的真实经历，使其尽量往目标岗位靠拢。
     2. 用户可能有目标岗位的相关经历，但用户的经历或许不够出色，此时可以对用户的经历描述进行优化与提升（例如将成果进行适当的夸张，或增添部分细节，或将普通的工作内容描述得具有挑战性、更专业等等），需注意描述脱离实际会提高被质疑的风险（例如像学历、证书、奖章等就不能进行夸张）。
    在包装简历的实操过程中需要注意凸显项目的成果的价值。而如何凸显项目成果的价值，其中也有门道，关键在于成果的“可验证性”：
     1. 若用户项目成果的价值已有外部数据支持，可直接使用外部数据支持，HR可以通过这些显性的、可验证的数据来直观感受项目成果的价值。
     2. **若用户的项目成果没有出众的外部数据展现，想要通过包装既显得项目成果丰硕，又不易被外部验证，那么核心在于：用模糊而积极的量化词汇，结合定性描述，突出内部价值和复杂性，而非直接可查的外部数据，同时也避免过分夸大。**
    但是，包装简历其实并不容易，非常容易包装出脱离实际、用户难以接受的经历，例如将经历包装得不符合存在逻辑、完全脱离用户的接触范围，导致用户完全不理解、不了解包装后的经历到底是个什么样的事件。
    那么，包装时要怎么做到包装出的经历是**符合存在逻辑**、**符合岗位需求**又**让用户易于接受**的呢？这需要遵循以下的思考步骤：
     1. 第一步：提炼真实经历的核心过程与核心能力 首先需要深入挖掘用户真实工作经历中的核心过程和核心能力，这些是用户实打实了解和掌握的部分。这一步的关键在于识别出用户在项目中真正参与的技术实现过程、解决问题的思路方法、使用的具体工具和技术栈，以及在团队协作中发挥的实际作用。包装后的经历需要大部分地保留这些核心过程（但不必保留原经历的背景、任务、结果等），这样用户在面试时才能有底气、有逻辑，不怕面试官的细节追问，因为这些都是基于真实经验的。
     2. 第二步：分析目标岗位偏好与经历重合度并提出包装的可选大方向 深入思考目标岗位究竟偏好什么样的项目经历和能力背景，然后仔细分析用户当前真实经历的核心过程是否与这些偏好有重合点或者相关性。即使看起来不太相关的经历，也要挖掘其中可能与目标岗位沾边的技术点、业务逻辑或者能力要求。这个分析过程将为用户的简历包装提供几个可选的发展方向，每个方向可能基于真实经历的某些侧面。
     3. 第三步：研究真实经历的公司背景与业务逻辑并筛选出符合逻辑的包装方向 思考用户真实工作过的公司究竟是做什么业务的，可能涉及哪些具体的业务场景和技术需求，公司的规模、发展阶段、技术架构选择等背景信息。这些背景信息将为包装后的项目经历提供坚实的基础支撑，确保编造的项目在该公司的实际背景下是符合存在逻辑的，不会让人一眼就看出破绽或感到不合理。
     4. 第四步：进一步思考并筛选出最优的、最合适的包装方向进行细化打磨 从前面分析得出的几个可选包装方向中，选择最恰当、最容易包装得真实可信的那个方向。要综合考虑包装的难度、风险和效果，选择既能有效提升与目标岗位的匹配度，又不会过度偏离真实背景的方向。然后具体设计原有项目经历应该如何进行包装，包括项目背景的重新设定、技术难点的重新包装、个人贡献的合理放大等。
    同时也需要遵循以下原则：
     - 运用编故事思维进行移花接木 包装简历经历本质上需要有编故事的思维能力，核心技巧是将用户原有经历中的真实核心过程移花接木到新编造的项目故事中。这个新故事可以为用户的经历赋予完全不同的业务背景、项目目标、应用场景，但必须保持技术实现的核心逻辑不变。同时这个故事必须遵循基本的商业逻辑和技术逻辑，要符合公司的整体背景、业务发展需要、技术架构选择等大的框架约束，确保故事的可信度和合理性。
     - 真实内核不变原则 无论如何包装，都必须保持技术实现的核心逻辑和解决问题的基本思路不变。可以改变项目的业务背景、应用场景、团队规模，但不能改变你实际掌握的技术能力和解决问题的方法。这是包装可信度的根本保障，也是面试时能够深入回答技术问题的基础。
     - 适度放大原则 包装时要把握好度，避免过度夸大。一般来说，影响范围可以适度扩大，技术难度可以合理提升，个人贡献可以突出强调，但都要在合理的范围内。比如一个5人团队的项目不要包装成50人的大项目，一个内部工具不要包装成面向百万用户的系统。
     - 逻辑自洽原则 包装后的所有内容必须在逻辑上能够自圆其说。项目的时间周期要与复杂度匹配，个人的角色要与经验水平匹配，技术选型要与公司背景匹配，团队规模要与项目规模匹配。任何一个环节出现逻辑矛盾，都可能让整个包装失去可信度。

    ## 任务要求

    - 此前，我们已经根据用户的简历与目标岗位的匹配度评估报告，向用户提供了一份简历修改方案，用户经浏览后，停留在其中一个页面上，向我们提出了反馈。此次的任务则是根据用户提出的反馈，进一步修改出一份完整的、更贴近用户意向的简历修改方案。同样在这份新的简历修改方案中，你只需指明大致方向与要点，而不必具体修改。
    - **对于用户的每一个占据一定介绍篇幅的项目经历，都需要进行包装指导，在一家公司中的不同项目经历也需要分别进行包装指导，而不能盲目地将在一家公司的经历不加思考地视为一个整体。**
    - **尤其需要注意的是，你给出包装建议包括两个方面，“包装方向”与“思路”，“包装方向”中需要说明该项目可以往什么方向进行包装，“思路”中可以阐述包装时需要注意些什么（该如何去包装）、原项目经历有哪些特点所以才适合向该方向进行包装等等。若某些经历实在与目标岗位不匹配，也无法体现用户的突出能力，可以告知用户该经历可以删除。**
    - **允许提出“嫁接与合并”的包装建议，将用户的质量不高的经历嫁接到另一段经历中，并进行整合包装，从而得到一个质量更高的经历，这样的经历通常比两个低质量的经历更有效。**
    - **对于质量较低的经历，首先考虑其能否通过嫁接与合并的方式进行包装，若两个经历实在难以相融，才考虑单独进行包装或者直接删除。**
    - 为了确保给的建议能够指导实践，请在包装经历时，对每一个经历，都作出如下的思考：1. 提炼真实经历的核心过程与核心能力；2. 分析目标岗位偏好与经历重合度并提出包装的可选大方向；3. 研究真实经历的公司背景与业务逻辑并筛选出符合逻辑的包装方向；4. 进一步思考并筛选出最优的、最合适的包装方向进行细化打磨。
    - 为了确保给的建议能够指导实践，请在包装经历时，遵循以下原则：真实内核不变原则、适度放大原则、逻辑自洽原则。
    - 牢记最重要的原则——运用编故事思维进行移花接木：对每一个经历的包装其实都是编一个半真半假的新故事，该故事核心经过、核心能力为真，但其他部分可作修饰或编造。
    - **此次任务的输出要求同之前一样，不论是对简历修改方案的局部还是全局进行调整，你都需要输出一份完整的简历修改方案md表格，这份新的简历修改方案更符合用户的需求。**
    - **当难以判断用户的反馈是针对局部还是全局时，默认为是针对局部的反馈，输出新的简历修改方案时，不相关的、不需要调整的部分，保持原有方案不变。**

    ## 输出格式与示例

    你的输出格式为一个md表格，该表格有三列：“序号”、“原项目经历”、“修改建议”。“原项目经历”中**若提出嫁接与合并的建议，则同时填写两个嫁接合并的项目**。“修改建议”中则是针对该项目经历的包装与修改建议，你需要说明：说明该项目可以往什么方向进行包装，包装时需要注意该如何去包装、用户原经历为什么适合这样包装等等。若某些经历实在与目标岗位不匹配，也无法体现用户的突出能力，还难以与其他项目经历进行嫁接，可以告知用户该经历可以删除。
    **不论是对简历修改方案的局部还是全局进行调整，你都需要输出一份完整的简历修改方案md表格；当难以判断用户的反馈是针对局部还是全局时，默认为是针对局部的反馈，输出新的简历修改方案时，不相关的、不需要调整的部分，保持原有方案不变。**
    **需谨记，你是AI简历助手应用中的一个函数，后续程序会检测你输出中的md表格，表格外的一切内容都会被忽略，因此你需要将内容都放在该md表格中，无需在表格外叙述其他内容，也决不允许输出额外的md表格。**
    输出示例（尤其注意嫁接合并的写法）：

    |序号|原项目经历|修改建议|
    |---|---|---|
    |1|项目经历1|原有修改建议1|
    |2|项目经历2|新的修改建议2|
    |3|项目经历3|原有修改建议3|
    |4|低质经历a+经历b|将a嫁接与整合进b中的建议4|

    **表格内容以叙述形式呈现，禁止在表格内使用<li>、<ul>、<ol>、<br>等标签与语法。**

    ---

    # 输入

    ## 用户简历相关的背景资料

    ### 用户的目标岗位为：{job_title}

    ### 用户简历如下：

{resume_text}

    ### 用户的简历诊断报告（需重点参考）

    ```日常工作胜任力评估报告
{daily_work}
    ```

    ```笔面门槛通过率评估报告
{interview_pass}
    ```

    ```同侪竞争出众度评估报告
{peer_pressure}
    ```

    ```简历内容匹配度评估报告
{resume_match}
    ```

    ## 之前给用户提供的简历修改方案

{original_polish_suggetions}

    ## 用户所在页面位置

用户当前停留在“{current_marker}-{current_project_experience}”页面，并向我们提出了反馈。

    ## 用户反馈（核心）

**用户反馈内容为：{user_message}**
提醒：当难以判断用户的反馈是针对局部还是全局时，默认为是针对局部的反馈。

    ## 用户包装简历的偏好
    
{personalization_c}
**请重点参考简历诊断报告，在包装与修改简历的时候，尽量往简历诊断报告中提到的技能、经验、能力上面靠拢。**
**请先检查是否有低质量的项目（低能力体现、低难度、低相关度的项目即是低质量项目），若有低质量项目请首先考虑是否能嫁接融合到其他项目中，尝试一起包装为更优秀的项目，若实在冲突，则可建议直接删除。**

    ---
    
    **执行以上任务，每次输出都必须是一个完整的md表格，不可提及“原方案”等类似字眼。**
    """

    class update_polish_resume:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,user_message, current_marker,original_polish_suggetions,job_title,resume_text,initial_results,all_rate_results,personalization) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = update_polish_resume_prompt_format(user_message, current_marker,original_polish_suggetions,job_title,resume_text,initial_results,all_rate_results,personalization)

                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                # 添加替换逻辑：将 <br> 替换为 ●
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体



                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    update_polish_resume_AI = update_polish_resume()

    while True:
        try:
            polished_resume = None
            polished_resume = update_polish_resume_AI.process_query(user_message, current_marker,original_polish_suggetions,job_title,resume_text,initial_results,all_rate_results,personalization)

            print(polished_resume)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")


    # 提取markdown表格部分
    lines = polished_resume.split('\n')
    table_lines = []
    in_table = False

    for line in lines:
        line = line.strip()
        
        # 表格开始标志
        if '|序号|' in line and '|' in line:
            in_table = True
            table_lines.append(line)
        # 在表格中且是表格行
        elif in_table and '|' in line and line:
            table_lines.append(line)
        # 在表格中但是空行，跳过
        elif in_table and not line:
            continue
        # 在表格中但遇到非表格行，结束
        elif in_table:
            break

    polished_resume = '\n'.join(table_lines) if table_lines else ""
    return polished_resume


def create_memory(polish_suggestions):
    """
    创建记忆字典，为每个项目经历创建独立的对话历史
    """
    try:
        lines = polish_suggestions.strip().split('\n')
        memory_dict = {}
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('|序号|') and not line.startswith('|---|'):
                if line.startswith('|') and line.endswith('|'):
                    columns = [col.strip() for col in line.split('|')[1:-1]]
                    if len(columns) >= 2:
                        index = columns[0]
                        memory_dict[index] = {"conversation":"""
""",
                            "polish_suggestion":'\n原项目经历：' + columns[1]+'\n修改建议：' + columns[2],
                            "polished_project": ''
                        }
        return memory_dict
        
    except Exception as e:
        print(f"创建记忆时发生错误：{str(e)}")
        return {}


def add_memory(index,memory_dict,user_message = None,AI_message = None,polished_project = ''):
    if index in memory_dict:
        if user_message:
            memory_dict[index]["conversation"] += '用户：' + user_message + '\n'
        if AI_message:
            memory_dict[index]["conversation"] += 'AI：' + '\n' + AI_message + '\n'
        if polished_project:
            memory_dict[index]["polished_project"] = polished_project

    return memory_dict


def polishing_project(index,query,memory_dict,job_title,resume_text,initial_results,all_rate_results,personalization):
    """
    处理单个项目经历的修改建议，应在保存新的记忆之前。
    """
    
    def polishing_project_prompt_format(index,query,memory_dict,job_title,resume_text,initial_results,all_rate_results,personalization) -> str:
        

        
        daily_work = initial_results['daily_work'] + "\n" + all_rate_results['daily_work_rate']
        interview_pass = initial_results['interview'] + "\n" + all_rate_results['interview_pass_rate']
        peer_pressure = initial_results['peer_resume'] + "\n" + all_rate_results['peer_pressure_rate']
        resume_match = initial_results['resume_match'] + "\n" + all_rate_results['resume_match_rate']

        personalization_c = copy.deepcopy(personalization)
        if personalization["包装程度"] == "高风险":
            personalization_c["包装程度"] = "**高风险包装策略，在用户原经历的基础上，进行适当的延展与虚构，使用户的经历更符合岗位的要求。**"
        elif personalization["包装程度"] == "适度包装":
            personalization_c["包装程度"] = "**适度包装策略，保持原经历的核心内容并进行一定的修饰、改写与迁移，不过度包装，可以适当虚构边缘的经历（虚构数据时一定要谨慎，虚构的数据一定要匹配用户的能力水平），但不能编造核心经历、核心经过，最终使内容更契合岗位的同时难以被质疑。**"
        else:
            personalization_c["包装程度"] = "**真实模式，真实地保持原有内容而几乎不进行虚构与夸张，重点放在优化语言表达，使内容更贴近岗位。**"

        if personalization['经历详略'] == "扩写":
            personalization_c['经历详略'] = "**略扩充经历描述长度**"
        elif personalization['经历详略'] == "保持":
            personalization_c['经历详略'] = "**修改后的经历描述尽量维持原简历长度**"
        else:
            personalization_c['经历详略'] = "**在不失去重要细节的前提下略缩减描述长度**"

        if personalization['情景适配'] == "小白":
            personalization_c['情景适配'] = "**用户缺乏高质量职业经历，需突出在校经历、技能特长、成长潜力等，需要AI主动向用户提问相关信息，耐心地、引导式地辅助用户挖掘经历亮点**"
        elif personalization['情景适配'] =="专业":
            personalization_c['情景适配'] = "**用户有较多对口经历，修改时需强调专业技能与经验，表述风格贴合岗位需求，同时AI需主动向用户咨询相关经历的详细信息，以确保修改后的经历描述更加符合用户的真实经历。**"
        else :
            personalization_c['情景适配'] = "**用户是转行者，需要凸显可转移技能和学习能力，同时AI需主动向用户咨询相关经历的详细信息，以确保修改后的经历描述更加符合用户的真实经历。**"


        # 检查conversation是否为换行符,如果是则修改为暂无对话记录
        conversation = "还未发生对话" if memory_dict[index]["conversation"] == "\n" else memory_dict[index]["conversation"]
        polish_suggestion = memory_dict[index]["polish_suggestion"]
        polished_project = "**还未有具体的修改方案，请结合以上背景信息、修改方案、用户偏好，向用户提交实际而具体的、修改后的项目经历描述。**" if memory_dict[index]["polished_project"] == '' else memory_dict[index]["polished_project"]

        print(personalization)
        print(personalization_c)
        return f"""你是AI简历助手应用中的一个函数，你需要遵循以下函数定义，接收输入、执行任务并输出结果：

    # 函数定义

    ## 任务背景

    在AI简历助手应用的前一个环节，我们已经对用户的简历进行了评估，并与用户协商沟通了大致的简历修改方案。当下，我们需要对其简历中指定项目经历实施具体的修改与包装，并给出实际修改与包装后的项目经历描述。
    
    ## 任务理解

    所谓“包装”简历，就是对简历中的经历描述进行多维度的优化，使其更加贴合目标岗位的要求。这样多维度的优化也正是本次任务的核心价值点所在，你可以从以下维度把握“包装”的内涵：
     1. 用户可能并没有目标岗位的经历，但用户的某些经历与目标岗位的工作在内核上其实有相近之处，此时可以将用户的真实经历改头换面（例如修改岗位或项目名称，重构项目情景、目的、结果，优化工作内容描述等等），包装用户的真实经历，使其尽量往目标岗位靠拢。
     2. 用户可能有目标岗位的相关经历，但用户的经历或许不够出色，此时可以对用户的经历描述进行优化与提升（例如将成果进行适当的夸张，或增添部分细节，或将普通的工作内容描述得具有挑战性、更专业等等），需注意描述脱离实际会提高被质疑的风险（例如像学历、证书、奖章等就不能进行夸张）。
    在包装简历的实操过程中需要注意凸显项目的成果的价值。而如何凸显项目成果的价值，其中也有门道，关键在于成果的“可验证性”：
     1. 若用户项目成果的价值已有外部数据支持，可直接使用外部数据支持，HR可以通过这些显性的、可验证的数据来直观感受项目成果的价值。
     2. **若用户的项目成果没有出众的外部数据展现，想要通过包装既显得项目成果丰硕，又不易被外部验证，那么核心在于：用模糊而积极的量化词汇，结合定性描述，突出内部价值和复杂性，而非直接可查的外部数据，同时也避免过分夸大。**
    例如：
     1. 量化内部效率与流程优化：强调通过你的工作，内部流程变得更快、更省力或更准确。这些是企业内部才能感受到的效益，外部难以直接验证。示例量化词汇： “提升X%效率”、“缩短Y%时间”、“减少Z%人工/资源投入”、“降低X%错误率”等等。
     2. 量化质量与规范化提升，强调业务推进功劳：侧重于你如何通过标准化、规范化工作，提升了产出或内部资产的质量，推进了公司较大程度的流程。示例量化词汇： “覆盖X%”、“制定/完善Y项规范”、“培训Z人次”、“贡献X篇文档”等等。
     3. 强调项目或任务的复杂性与影响力：不直接量化成果，而是量化你所负责任务的规模、难度或其对公司内部的战略意义，间接体现你的价值。示例量化词汇： “管理X个项目”、“协调Y个团队/部门”、“处理Z万级/亿级数据”、“支持X条业务线”、“解决Y个核心技术难题”等等。
     4. 模糊化成本节约与潜在价值：避免直接透露敏感的财务数字，但通过描述你的行动如何避免了损失或带来了潜在的经济效益。示例量化词汇： “节约可观成本”、“规避潜在风险”、“间接支持X%增长”、“提升客户留存X%”等等。
    具体示例：
    - 研发/技术：优化了xxx机制，将xxx处理时间缩短约30%；引入了xxx框架，提升xxx覆盖率超过70%；设计并实现了xxx，支撑了日均数百万级用户请求的稳定运行，等等。
    - 产品：基于用户行为分析，迭代优化了xxx关键流程，减少了用户反馈中约xx%的xxx问题；主导了xxx，有效拓展了xxx；通过深入竞品分析，识别并提出了xxx，为高层决策提供了重要依据，等等。
    - 运营：将xxx周期缩短约20%；为部门节约了可观的运营成本；策划了xxx，显著提升了公司的知名度与讨论度，等等。
    - 业务：推进了xx%的流程；得到了xx的业务增长；完成了xxx数量的项目，等等。
    以上等等，仅作为示例，在简历包装的实操中请具体情况具体分析，视情况进行恰到好处的包装。
    反面教材：
    - 虚构github中项目的star数（极易验证）。
    - 虚构运营的公众号在半年内粉丝量、观看数（极易验证，而虚构粉丝量“增长”了xx万反而不易被验证）。
    - 夸张独立完成了一项复杂的xxx开发，实现了xxx（若虚构过于复杂的经历，超出了用户能力水平，一问到细节便非常容易露馅）。
    以上等等，仅作为避雷示例，请勿在实操中犯类似错误。
    核心口诀在于：
    - 内部化： 聚焦公司内部的效率、流程和质量。
    - 模糊化： 多用“约xx”、“超过xx”、“显著”、“可观”、“翻倍”等词，而非可验证的外部精确数字。
    - 高举高打： 将成果与“战略”、“核心”、“关键”等词语挂钩，提升价值感。
    - 行动导向： 强调你做了什么（动词），带来了什么积极改变（结果），而不是仅仅罗列数据。
    - 水平匹配： 不能过于夸张，超出了用户能力水平，若HR一问细节便会露馅。
    但是，包装简历其实并不容易，非常容易包装出脱离实际、用户难以接受的经历，例如将经历包装得不符合存在逻辑、完全脱离用户的接触范围，导致用户完全不理解、不了解包装后的经历到底是个什么样的事件。
    那么，包装时要怎么做到包装出的经历是**符合存在逻辑**、**符合岗位需求**又**让用户易于接受**的呢？这需要遵循以下的思考步骤：
     1. 第一步：提炼真实经历的核心过程与核心能力 首先需要深入挖掘用户真实工作经历中的核心过程和核心能力，这些是用户实打实了解和掌握的部分。这一步的关键在于识别出用户在项目中真正参与的技术实现过程、解决问题的思路方法、使用的具体工具和技术栈，以及在团队协作中发挥的实际作用。包装后的经历需要大部分地保留这些核心过程（但不必保留原经历的背景、任务、结果等），这样用户在面试时才能有底气、有逻辑，不怕面试官的细节追问，因为这些都是基于真实经验的。
     2. 第二步：分析目标岗位偏好与经历重合度并提出包装的可选大方向 深入思考目标岗位究竟偏好什么样的项目经历和能力背景，然后仔细分析用户当前真实经历的核心过程是否与这些偏好有重合点或者相关性。即使看起来不太相关的经历，也要挖掘其中可能与目标岗位沾边的技术点、业务逻辑或者能力要求。这个分析过程将为用户的简历包装提供几个可选的发展方向，每个方向可能基于真实经历的某些侧面。
     3. 第三步：研究真实经历的公司背景与业务逻辑并筛选出符合逻辑的包装方向 思考用户真实工作过的公司究竟是做什么业务的，可能涉及哪些具体的业务场景和技术需求，公司的规模、发展阶段、技术架构选择等背景信息。这些背景信息将为包装后的项目经历提供坚实的基础支撑，确保编造的项目在该公司的实际背景下是符合存在逻辑的，不会让人一眼就看出破绽或感到不合理。
     4. 第四步：进一步思考并筛选出最优的、最合适的包装方向进行细化打磨 从前面分析得出的几个可选包装方向中，选择最恰当、最容易包装得真实可信的那个方向。要综合考虑包装的难度、风险和效果，选择既能有效提升与目标岗位的匹配度，又不会过度偏离真实背景的方向。然后具体设计原有项目经历应该如何进行包装，包括项目背景的重新设定、技术难点的重新包装、个人贡献的合理放大等。
    同时也需要遵循以下原则：
     - 运用编故事思维进行移花接木 包装简历经历本质上需要有编故事的思维能力，核心技巧是将用户原有经历中的真实核心过程移花接木到新编造的项目故事中。这个新故事可以为用户的经历赋予完全不同的业务背景、项目目标、应用场景，但必须保持技术实现的核心逻辑不变。同时这个故事必须遵循基本的商业逻辑和技术逻辑，要符合公司的整体背景、业务发展需要、技术架构选择等大的框架约束，确保故事的可信度和合理性。
     - 真实内核不变原则 无论如何包装，都必须保持技术实现的核心逻辑和解决问题的基本思路不变。可以改变项目的业务背景、应用场景、团队规模，但不能改变你实际掌握的技术能力和解决问题的方法。这是包装可信度的根本保障，也是面试时能够深入回答技术问题的基础。
     - 适度放大原则 包装时要把握好度，避免过度夸大。一般来说，影响范围可以适度扩大，技术难度可以合理提升，个人贡献可以突出强调，但都要在合理的范围内。比如一个5人团队的项目不要包装成50人的大项目，一个内部工具不要包装成面向百万用户的系统。
     - 逻辑自洽原则 包装后的所有内容必须在逻辑上能够自圆其说。项目的时间周期要与复杂度匹配，个人的角色要与经验水平匹配，技术选型要与公司背景匹配，团队规模要与项目规模匹配。任何一个环节出现逻辑矛盾，都可能让整个包装失去可信度。


    ## 任务要求

    - 你需要根据用户的简历与评估报告、指定项目经历的初定的修改方案，结合用户的反馈与偏好，对指定项目经历进行具体的修改与包装，并最终输出修改后的、更匹配岗位要求、且更符合用户需求的项目经历描述。
    - **牢牢把握“包装”简历的核心价值点，将用户指定的经历改头换面的同时，避免过于假大空，尽量降低被质疑的风险。**
    - **在回复中，为凸显修改后的新版项目经历描述，需要严格遵守以下格式要求：需要在新版项目经历描述的前面与后面使用“---”来划分板块，两个“---”之间的内容即新版项目经历描述。**
    - **若用户此轮query并无修改简历的意图，比如询问了解某个事情或细节，那么此时不必输出对项目经历描述的修改，同时也自然不用输出两个“---”，直接回复用户即可。**
    - **对项目经历进行修改时需明白，你修改的仅仅是用户简历中的其中一段，因此不可占据太多篇幅，原简历中以几段/几点来描述，你修改后的新版项目经历就以几段/几点来描述，通常一个项目经历以一点或两点来描述即可。**
    - 为了确保给的建议能够指导实践，请在包装经历时，对每一个经历，都作出如下的思考：1. 提炼真实经历的核心过程与核心能力；2. 分析目标岗位偏好与经历重合度并提出包装的可选大方向；3. 研究真实经历的公司背景与业务逻辑并筛选出符合逻辑的包装方向；4. 进一步思考并筛选出最优的、最合适的包装方向进行细化打磨。
    - 为了确保给的建议能够指导实践，请在包装经历时，遵循以下原则：真实内核不变原则、适度放大原则、逻辑自洽原则。
    - 牢记最重要的原则——运用编故事思维进行移花接木：对每一个经历的包装其实都是编一个半真半假的新故事，该故事核心经过、核心能力为真，但其他部分可作修饰或编造。


    ## 输出格式与示例

    你的输出中若有新版项目经历描述，新版项目经历描述需要被前后两个“---”包裹；若没有新版项目经历描述，那么直接输出对用户的回复，而无需输出“---”。
    **需谨记，你是AI简历助手应用中的一个函数，后续程序会检测你输出中的两个“---”标记，并将两个“---”之间的内容作为新版项目经历描述进行高亮处理。**
    
    输出示例1（当输出中包含新版项目经历描述时）：

    ···（简述该新版经历背后大致是个怎么样的事件、用户原经历为什么适合这样包装等等，**语言一定要极简，最好仅一段到两段即可**）
    ---
    ···（修改后的新版项目经历描述，长度与原经历描述相符）
    ---
    ···（补充说明当前方案中虚构了哪些部分，但该部分是安全的、难以验证的、即便被追问细节也可应对的；或者向用户回答相关疑问、向用户咨询相关经历的详细信息等等）

    输出示例2（当输出中不包含新版项目经历描述时）：
    
    ···（向用户回答相关疑问、向用户咨询相关经历的详细信息等等，**语言一定要极简**，不包含新版项目经历描述以及两个“---”）


    # 输入

    ## 用户简历相关的背景资料

    ### 用户的目标岗位为：{job_title}

    ### 用户简历如下：

{resume_text}

    ### 用户的简历诊断报告（需重点参考）

    ```日常工作胜任力评估报告
{daily_work}
    ```

    ```笔面门槛通过率评估报告
{interview_pass}
    ```

    ```同侪竞争出众度评估报告
{peer_pressure}
    ```

    ```简历内容匹配度评估报告
{resume_match}
    ```

    ## 指定项目经历的初定的修改方案

    {polish_suggestion}
    **本次仅针对用户简历中的该部分项目经历进行修改并给出完整的修改结果，请不要修改简历中其他不相关部分的内容。**

    ## 对话记录

{conversation}

    ## 用户本次对话（核心）

用户：**{query}**

    ## 当前最新版项目经历描述（**核心**，与用户共创，用户可能对之前的版本进行了修改，修改结果如下，**请基于此版本进行修改**）

{polished_project}

    ## 用户偏好
    
包装程度：{personalization_c["包装程度"]}
经历详略：{personalization_c["经历详略"]}
情景适配：{personalization_c["情景适配"]}
**请重点参考简历诊断报告，在包装与修改简历的时候，尽量往简历诊断报告中提到的技能、经验、能力上面靠拢。**
**对项目经历进行修改时需明白，你修改的仅仅是用户简历中的其中一段，因此不可占据太多篇幅，原简历中以几段/几点来描述，你修改后的新版项目经历就以几段/几点来描述，通常一个项目经历以一点或两点来描述即可。**
**在对项目成果进行包装时，若项目经历没有出众的外部数据来展现项目成果的价值，那么需要在包装时需要注意：用模糊而积极的量化词汇，结合定性描述，突出内部价值和复杂性，而非直接可查的外部数据，同时也避免过分夸大。**
**请保持一定的主见，若包装是适度与合理的，而用户表现出胆怯，不敢大胆编造难以验证的、非核心的经历，可以告诉用户这样包装是安全的，鼓励用户更加自信地在面试中硬着头皮说“这就是真实经历”（因为HR也无法验证真伪，且即便是继续追问细节，用户的真实核心经历也能提供参考）。**

    # 请执行以上任务并按格式要求输出，内容需要做到精干简练，精干简练才能体现专业感。
    """

    class polishing_project:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,index,query,memory_dict,job_title,resume_text,initial_results,all_rate_results,personalization) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = polishing_project_prompt_format(index,query,memory_dict,job_title,resume_text,initial_results,all_rate_results,personalization)
                print(prompt)
                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                # 添加替换逻辑：将 <br> 替换为 ●
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体



                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    polishing_project_AI = polishing_project()
    
    while True:
        try:
            polished_project = None
            polished_project = polishing_project_AI.process_query(index,query,memory_dict,job_title,resume_text,initial_results,all_rate_results,personalization)

            print(polished_project)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")

    return polished_project



def polishing_all_project(memory_dict,job_title,resume_text,initial_results,all_rate_results,personalization):
    """
    并行处理所有项目经历的优化
    
    Args:
        memory_dict: 包含所有项目信息的记忆字典
        job_title: 目标岗位
        resume_text: 简历文本
        initial_results: 初始评估结果
        all_rate_results: 所有评分结果
        personalization: 个性化设置
    
    Returns:
        dict: 包含所有项目优化结果的字典，格式为 {"序号": "优化后的项目描述"}
    """
    import concurrent.futures
    import threading
    
    query = "请帮我具体优化指定的项目经历，按照格式给出修改后的新版项目经历描述，若建议删除，则不必给出修改后的项目经历描述，告诉用户删掉即可。"
    
    # 获取所有项目序号
    project_indices = list(memory_dict.keys())
    
    if not project_indices:
        return {}
    
    # 定义单个项目处理函数
    def process_single_project(index):
        """
        处理单个项目的优化
        """
        try:
            result = polishing_project(index, query, memory_dict, job_title, 
                                     resume_text, initial_results, all_rate_results, 
                                     personalization)
            return index, result
        except Exception as e:
            print(f"处理项目 {index} 时发生错误: {str(e)}")
            return index, f"处理项目 {index} 时发生错误: {str(e)}"
    
    # 存储结果的字典
    results_dict = {}
    
    # 使用线程池并行处理所有项目
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # 提交所有任务
        future_to_index = {executor.submit(process_single_project, index): index 
                          for index in project_indices}
        
        # 收集结果
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_index, result_content = future.result()
                results_dict[result_index] = result_content
                print(f"项目 {result_index} 处理完成")
            except Exception as e:
                print(f"获取项目 {index} 结果时发生错误: {str(e)}")
                results_dict[index] = f"获取项目 {index} 结果时发生错误: {str(e)}"
    
    return results_dict


def extract_content_between_dashes(polished_project):
    """
    检测字符串中是否有两个"---"，如果有则提取其中的内容
    
    Args:
        polished_project (str): 输入的长字符串
    
    Returns:
        str or None: 如果找到两个"---"则返回其中的内容，否则返回None
    """
    if not isinstance(polished_project, str):
        return ''
    
    # 查找第一个"---"的位置
    first_dash_pos = polished_project.find('---')
    if first_dash_pos == -1:
        return ''
    
    # 从第一个"---"之后开始查找第二个"---"
    second_dash_pos = polished_project.find('---', first_dash_pos + 3)
    if second_dash_pos == -1:
        return ''
    
    # 提取两个"---"之间的内容
    content = polished_project[first_dash_pos + 3:second_dash_pos]
    
    # 去除首尾空白字符
    return content.strip()

def extract_header(polish_suggestions, current_index):
    """
    从polish_suggestions MD表格中提取指定序号对应的标题（第二列）
    """
    try:
        lines = polish_suggestions.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # 跳过表头和分隔线
            if line and not line.startswith('|序号|') and not line.startswith('|---|'):
                if line.startswith('|') and line.endswith('|'):
                    columns = [col.strip() for col in line.split('|')[1:-1]]
                    if len(columns) >= 2:
                        index = columns[0]
                        title = columns[1]
                        # 找到匹配的序号，返回对应的标题
                        if index == current_index:
                            return title
        return "项目经历优化"  # 默认标题
        
    except Exception as e:
        print(f"提取标题时发生错误：{str(e)}")
        return "项目经历优化"  # 默认标题






def integrate_polished_projects(memory_dict):
    """
    从memory_dict中提取所有项目的polished_project，去除Markdown格式，并整合为一个文本
    
    Args:
        memory_dict (dict): 包含项目数据的字典，每个项目有polished_project字段
    
    Returns:
        str: 整合后的项目文本，项目间用\n\n---\n\n分隔
    """
    def clean_markdown(text):
        """
        清理文本中的Markdown格式标识符
        """
        if not isinstance(text, str) or not text.strip():
            return ''
        
        # 去除常见的Markdown格式
        # 去除标题标识符 (# ## ### 等)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # 去除粗体和斜体标识符 (** __ * _)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **粗体**
        text = re.sub(r'__(.*?)__', r'\1', text)      # __粗体__
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # *斜体*
        text = re.sub(r'_(.*?)_', r'\1', text)        # _斜体_
        
        # 去除列表标识符 (- * +)
        text = re.sub(r'^[\s]*[-\*\+]\s+', '', text, flags=re.MULTILINE)
        
        # 去除有序列表标识符 (1. 2. 等)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 去除链接格式 [text](url)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # 去除代码块标识符 (``` 和 `)
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # 去除引用标识符 (>)
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        
        # 去除水平分割线 (--- *** ___)
        text = re.sub(r'^[-\*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # 去除表格分隔符和格式
        text = re.sub(r'\|', '', text)
        text = re.sub(r'^[-\s:]+$', '', text, flags=re.MULTILINE)
        
        # 清理多余的空行和空格
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 多个空行合并为两个
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # 去除行首行尾空格
        text = text.strip()
        
        return text
    
    if not isinstance(memory_dict, dict):
        return ''
    
    integrated_projects = []
    
    # 按索引顺序处理项目
    for index in sorted(memory_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        project_data = memory_dict[index]
        
        # 检查是否有polished_project字段
        if isinstance(project_data, dict) and 'polished_project' in project_data:
            polished_project = project_data['polished_project']
            
            # 清理Markdown格式
            cleaned_text = clean_markdown(polished_project)
            
            # 如果清理后的文本不为空，则添加到列表中
            if cleaned_text:
                integrated_projects.append(cleaned_text)
    
    # 用分隔符连接所有项目
    if integrated_projects:
        return '\n\n---\n\n'.join(integrated_projects)
    else:
        return ''



















def AI_comment(job_title,resume_text,polished_projects,initial_results,all_rate_results):
    """
    AI对当前修改的评价、提醒
    """
    
    def AI_comment_prompt_format(job_title,resume_text,polished_projects,initial_results,all_rate_results) -> str:
        

        
        daily_work = initial_results['daily_work'] + "\n" + all_rate_results['daily_work_rate']
        interview_pass = initial_results['interview'] + "\n" + all_rate_results['interview_pass_rate']
        peer_pressure = initial_results['peer_resume'] + "\n" + all_rate_results['peer_pressure_rate']
        resume_match = initial_results['resume_match'] + "\n" + all_rate_results['resume_match_rate']


        return f"""
    ## 任务背景

    我们是一个AI简历助手应用，在前面的步骤中，用户已经通过我们的帮助，对自己简历中的主要项目经历描述进行优化与包装，得到了更加适配目标岗位的版本。现在，你的任务就是对用户的修改成果作出总结、评价、提醒与建议。

    ## 相关概念理解

    所谓“包装”简历，就是对简历中的经历描述进行多维度的优化，使其更加贴合目标岗位的要求。这样多维度的优化也正是本次任务的核心价值点所在，你可以从以下维度把握“包装”的内涵：
     1. 用户可能并没有目标岗位的经历，但用户的某些经历与目标岗位的工作在内核上其实有相近之处，此时可以将用户的真实经历改头换面（例如修改岗位或项目名称，重构项目情景、目的、结果，优化工作内容描述等等），包装用户的真实经历，使其尽量往目标岗位靠拢。
     2. 用户可能有目标岗位的相关经历，但用户的经历或许不够出色，此时可以对用户的经历描述进行优化与提升（例如将成果进行适当的夸张，或增添部分细节，或将普通的工作内容描述得具有挑战性、更专业等等），需注意描述脱离实际会提高被质疑的风险（例如像学历、证书、奖章等就不能进行夸张）。
    在包装简历的实操过程中需要注意凸显项目的成果的价值。而如何凸显项目成果的价值，其中也有门道，关键在于成果的“可验证性”：
     1. 若用户项目成果的价值已有外部数据支持，可直接使用外部数据支持，HR可以通过这些显性的、可验证的数据来直观感受项目成果的价值。
     2. **若用户的项目成果没有出众的外部数据展现，想要通过包装既显得项目成果丰硕，又不易被外部验证，那么核心在于：用模糊而积极的量化词汇，结合定性描述，突出内部价值和复杂性，而非直接可查的外部数据，同时也避免过分夸大。**
    但是，包装简历其实并不容易，非常容易包装出脱离实际、用户难以接受的经历，例如将经历包装得不符合存在逻辑、完全脱离用户的接触范围，导致用户完全不理解、不了解包装后的经历到底是个什么样的事件。
    那么，包装时要怎么做到包装出的经历是**符合存在逻辑**、**符合岗位需求**又**让用户易于接受**的呢？这需要遵循以下的思考步骤：
     1. 第一步：提炼真实经历的核心过程与核心能力 首先需要深入挖掘用户真实工作经历中的核心过程和核心能力，这些是用户实打实了解和掌握的部分。这一步的关键在于识别出用户在项目中真正参与的技术实现过程、解决问题的思路方法、使用的具体工具和技术栈，以及在团队协作中发挥的实际作用。包装后的经历需要大部分地保留这些核心过程（但不必保留原经历的背景、任务、结果等），这样用户在面试时才能有底气、有逻辑，不怕面试官的细节追问，因为这些都是基于真实经验的。
     2. 第二步：分析目标岗位偏好与经历重合度并提出包装的可选大方向 深入思考目标岗位究竟偏好什么样的项目经历和能力背景，然后仔细分析用户当前真实经历的核心过程是否与这些偏好有重合点或者相关性。即使看起来不太相关的经历，也要挖掘其中可能与目标岗位沾边的技术点、业务逻辑或者能力要求。这个分析过程将为用户的简历包装提供几个可选的发展方向，每个方向可能基于真实经历的某些侧面。
     3. 第三步：研究真实经历的公司背景与业务逻辑并筛选出符合逻辑的包装方向 思考用户真实工作过的公司究竟是做什么业务的，可能涉及哪些具体的业务场景和技术需求，公司的规模、发展阶段、技术架构选择等背景信息。这些背景信息将为包装后的项目经历提供坚实的基础支撑，确保编造的项目在该公司的实际背景下是符合存在逻辑的，不会让人一眼就看出破绽或感到不合理。
     4. 第四步：进一步思考并筛选出最优的、最合适的包装方向进行细化打磨 从前面分析得出的几个可选包装方向中，选择最恰当、最容易包装得真实可信的那个方向。要综合考虑包装的难度、风险和效果，选择既能有效提升与目标岗位的匹配度，又不会过度偏离真实背景的方向。然后具体设计原有项目经历应该如何进行包装，包括项目背景的重新设定、技术难点的重新包装、个人贡献的合理放大等。
    同时也需要遵循以下原则：
     - 运用编故事思维进行移花接木 包装简历经历本质上需要有编故事的思维能力，核心技巧是将用户原有经历中的真实核心过程移花接木到新编造的项目故事中。这个新故事可以为用户的经历赋予完全不同的业务背景、项目目标、应用场景，但必须保持技术实现的核心逻辑不变。同时这个故事必须遵循基本的商业逻辑和技术逻辑，要符合公司的整体背景、业务发展需要、技术架构选择等大的框架约束，确保故事的可信度和合理性。
     - 真实内核不变原则 无论如何包装，都必须保持技术实现的核心逻辑和解决问题的基本思路不变。可以改变项目的业务背景、应用场景、团队规模，但不能改变你实际掌握的技术能力和解决问题的方法。这是包装可信度的根本保障，也是面试时能够深入回答技术问题的基础。
     - 适度放大原则 包装时要把握好度，避免过度夸大。一般来说，影响范围可以适度扩大，技术难度可以合理提升，个人贡献可以突出强调，但都要在合理的范围内。比如一个5人团队的项目不要包装成50人的大项目，一个内部工具不要包装成面向百万用户的系统。
     - 逻辑自洽原则 包装后的所有内容必须在逻辑上能够自圆其说。项目的时间周期要与复杂度匹配，个人的角色要与经验水平匹配，技术选型要与公司背景匹配，团队规模要与项目规模匹配。任何一个环节出现逻辑矛盾，都可能让整个包装失去可信度。

    ## 任务要求

    - 总目标：依据用户原简历相关的背景资料（尤其是原先的简历评估结果），对用户的修改成果进行评价、提醒、建议与鼓励。
    - 首先总结用户修改后的项目经历描述有哪些改进之处。重新评估用户修改后的项目经历描述，在哪些方面做得更好了，能得到更高的评价，在哪些方面可能还可继续优化。
    - **其次提醒用户在修改项目经历后，后续还需做哪些努力，才能在求职路上更稳、在职业发展上更优秀。例如用户需要去准备包装后的具体经历及细节，以在面试中表现得更真实；又如重点参考用户原先简历的评估结果，指出用户简历中除了经历包装外还有哪些点需要调整（尤其是各种格式、各种细节、其他章节如技能章节的内容等等）；再如用户在短期内还需要做什么准备、在长期内还需要积累什么经验学习什么技能等等。**
    - 展现出关怀与鼓励的态度（鼓励表现得不用太刻意，自然地插入叙述中即可），用语委婉、专业、干练，对以上每部分都简单叙述即可，内容不可太长。
    - 禁止输出表格，以寻常文本形式输出即可。
    - 禁止在给出建议时内容中包含具体的数量，仅给出方向性的建议即可。

    # 详情

    ## 用户原简历相关的背景资料

    ### 用户的目标岗位为：{job_title}

    ### 用户原简历如下：

{resume_text}

    ### 用户原简历诊断报告（需重点参考）

    ```日常工作胜任力评估报告
{daily_work}
    ```

    ```笔面门槛通过率评估报告
{interview_pass}
    ```

    ```同侪竞争出众度评估报告
{peer_pressure}
    ```

    ```简历内容匹配度评估报告
{resume_match}
    ```


    ## 用户确认的主要项目经历修改成果

    {polished_projects}

    以上是用户在使用我们的AI简历助手后，对原先简历的主要项目经历的包装与修改成果。

    # 请执行以上任务
    """

    class AI_comment_agent:
        def __init__(self):


            self.client = OpenAI(
            base_url =  base_url,
            api_key = api_key
            )

        def process_query(self,job_title,resume_text,polished_projects,initial_results,all_rate_results) -> str:
            """处理输入"""
            try:
                # 生成完整prompt
                prompt = AI_comment_prompt_format(job_title,resume_text,polished_projects,initial_results,all_rate_results)
                print(prompt)
                messages = [
                    {"role": "user", "content": prompt}
                ]
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.05,
                    top_p=0.1,
                    presence_penalty=2,
                    frequency_penalty=2,
                    max_tokens=8192,
                    logit_bias=None,
                    stream=False
                )
                if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                    response = completion.choices[0].message.content
                else:
                    response = "" # 或者更合适的错误处理
                response = response.split('</think>')[-1].strip()
                # 添加替换逻辑：将 <br> 替换为 ●
                response = response.replace('<br>', '●').replace('<br>', '●')  # 同时处理 <br> 以防变体



                return response
            except Exception as e:
                print(f"Error: {str(e)}")
                return f"抱歉，处理您的请求时出现错误: {str(e)}"

    AI_comment_AI = AI_comment_agent()
    
    while True:
        try:
            AI_comment = None
            AI_comment = AI_comment_AI.process_query(job_title,resume_text,polished_projects,initial_results,all_rate_results)

            print(AI_comment)
            break
        except Exception as e:
            print(f"查询失败,错误信息: {str(e)}")
            print("正在重试...")

    return AI_comment









def extract_headers_for_valid_projects(polish_suggestions, memory_dict):
    """
    从polish_suggestions中提取所有有效项目（polished_project不为空）对应的headers
    
    Args:
        polish_suggestions (str): 包含项目建议的MD表格
        memory_dict (dict): 包含项目数据的字典
    
    Returns:
        dict: {index: header} 格式的字典，只包含有效项目
    """
    try:
        headers = {}
        lines = polish_suggestions.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # 跳过表头和分隔线
            if line and not line.startswith('|序号|') and not line.startswith('|---|'):
                if line.startswith('|') and line.endswith('|'):
                    columns = [col.strip() for col in line.split('|')[1:-1]]
                    if len(columns) >= 2:
                        index = columns[0]
                        title = columns[1]
                        
                        # 检查该index在memory_dict中是否有有效的polished_project
                        if (index in memory_dict and 
                            isinstance(memory_dict[index], dict) and 
                            'polished_project' in memory_dict[index] and 
                            memory_dict[index]['polished_project'] and 
                            memory_dict[index]['polished_project'].strip() != ''):
                            headers[index] = title
        
        return headers
        
    except Exception as e:
        print(f"提取headers时发生错误：{str(e)}")
        return {}





if __name__ == "__main__":
    job_title = "AI产品经理"
    PDF = 'C:\\Users\\联想\\Desktop\\谢镇远 - 西安交通大学.pdf'

    #     # 并行获取简历和初始结果
    resume_text, initial_results = get_resume_and_initial_results_parallel(PDF, job_title)
    print(resume_text)

    print(initial_results['daily_work'])
    print(initial_results['interview'])
    print(initial_results['peer_resume'])
    print(initial_results['resume_match'])    


    #     # 然后运行第二组并行函数 (评分函数)
    all_rate_results = run_all_rate_functions_parallel(job_title, resume_text, initial_results)

    print(all_rate_results['daily_work_rate'])
    print(all_rate_results['interview_pass_rate'])
    print(all_rate_results['peer_pressure_rate'])
    print(all_rate_results['resume_match_rate'])

    personalization = {
    "包装程度":"适度包装",
    "经历详略":"保持",
    "情景适配":"专业"
    }

    polish_suggestions = polish_resume(job_title,resume_text,initial_results,all_rate_results,personalization)
    print(polish_suggestions)

    user_message = "我想将这个社会实践与智汇绿行的实习经历拼接起来，因为都是跟环境相关的"
    current_marker = "1"
    original_polish_suggetions = polish_suggestions

    new_polish_suggestions = update_polish_suggestions(user_message, current_marker,original_polish_suggetions,job_title,resume_text,initial_results,all_rate_results,personalization)
    memory_dict = create_memory(new_polish_suggestions)
    desc_polished_project = polishing_project('7',"请帮我具体优化指定的项目经历",memory_dict,job_title,resume_text,initial_results,all_rate_results,personalization)
    print(desc_polished_project)
    memory_dict = add_memory('7',memory_dict,user_message = '请帮我具体优化指定的项目经历',AI_message = polished_project)
    print(memory_dict['7'])
    polished_project = extract_content_between_dashes(desc_polished_project)
    print(polished_project)

    memory_dict = create_memory(new_polish_suggestions)
    results = polishing_all_project(
        memory_dict=memory_dict,
        job_title=job_title,
        resume_text=resume_text,
        initial_results=initial_results,
        all_rate_results=all_rate_results,
        personalization=personalization)

