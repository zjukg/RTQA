import json
import time
import asyncio
from openai import AsyncOpenAI
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
import os
import re
from dateutil import parser

question_path = "../../datasets/MultiTQ/questions/test.json"
with open(question_path, 'r') as file:
    raw_data = json.load(file)
#raw_data = [json.loads(line.strip()) for line in open('../../../released_data/2wikimultihopqa__v2_test_random_500.jsonl')]
q2sub_q = json.load(open("../TemQuesDecom/tree_full.json"))

trees = []

def normalize_date_in_question(question: str) -> str:
    """
    将英文自然语言中的时间表达转化为国际标准时间格式（YYYY-MM-DD 或 YYYY-MM）。
    """
    def convert_date(match):
        text = match.group(0)
        try:
            dt = parser.parse(text, fuzzy=True, default=None)
            # 判断是否包含日
            if re.search(r"\b\d{1,2}[a-z]{2}\b|\b\d{1,2}\b", text.lower()):
                return dt.strftime("%Y-%m-%d")
            elif any(month in text.lower() for month in MONTHS):
                return dt.strftime("%Y-%m")
            else:
                return dt.strftime("%Y")
        except Exception as e:
            return text  # 解析失败则返回原文

    # 支持的月份单词（缩写也匹配）
    MONTHS = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"
    ]

    # 匹配常见时间表达（先匹配更复杂的）
    patterns = [
        r"\b\d{1,2}(st|nd|rd|th)? (of )?(January|February|March|April|May|June|July|August|September|October|November|December)( \d{4})?\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(st|nd|rd|th)?,? \d{4}\b",
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{4}\b",
        r"\b\d{4}\b"
    ]

    for pattern in patterns:
        question = re.sub(pattern, convert_date, question, flags=re.IGNORECASE)

    return question


def dfs(q, tree):
    sons = []
    print(q)
    q = normalize_date_in_question(q)
    print(q)
    if not isinstance(q, str):
        idx = len(tree)
        tree.append({
            "idx": idx,
            "question_text": q,
            "sons": [],
            "qd_logprob": None
        })
        return idx
    # 取出子问题列表和打分
    sub_info = q2sub_q.get(q)
    if sub_info is None or not sub_info[0]:  # 没有找到 或 子问题列表为空
        # 原子问题，直接建一个节点
        idx = len(tree)
        tree.append({
            "idx": idx,
            "question_text": q,
            "sons": [],
            "qd_logprob": None
        })
        return idx

    # 如果有子问题，递归展开
    for sub_q in sub_info[0]:
        son_idx = dfs(sub_q, tree)
        sons.append(son_idx)

    idx = len(tree)
    tree.append({
        "idx": idx,
        "question_text": q,
        "sons": sons,
        "qd_logprob": q2sub_q.get(q, [[], None])[1]
    })    
    for son_idx in sons:
        tree[son_idx]["fa"] = idx
    return idx



for item in raw_data:
    #assert question in q2sub_q
    tree = []
    question = item['question'].strip()
    root_idx = dfs(question, tree)
    tree[root_idx]["gold_answer"] = item["answers"]
    tree[root_idx]["answer_type"] = item["answer_type"]
    tree[root_idx]["qlabel"] = item["qlabel"]
    tree[root_idx]["time_level"] = item["time_level"]
    tree[root_idx]["qtype"] = item["qtype"]
    trees.append(tree)

json.dump(trees, open("trees.json", "w"), indent=2)
    
    


    

