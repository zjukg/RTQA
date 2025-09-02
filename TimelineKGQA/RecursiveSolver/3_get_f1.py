import json
from tqdm import tqdm
from termcolor import colored
import math
from multiprocessing import Pool
from functools import partial
import argparse
import glob
import os
import re
import string
from sklearn.metrics import precision_score
import ast
from collections import defaultdict

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s)
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def eval_hit(prediction, answer):
    #print(f"Raw answer: {answer}")
    if answer is None:
        return 0
    if prediction is None:
        return 0
    
    
    
    #print(f"Raw answer: {answer}")
    
    if match(prediction, answer):
        return 1
    return 0
def topk(prediction, k=-1):
    if isinstance(prediction, list):
        return prediction[:k]
    elif isinstance(prediction, str):
        return [prediction]
    elif isinstance(prediction, int) or isinstance(prediction, float):
        return [str(prediction)]  # 转成字符串列表
    else:
        raise ValueError(f"Unsupported prediction type: {type(prediction)}")

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def normalize_prediction(prediction):
    if isinstance(prediction, list):
        return prediction
    elif isinstance(prediction, (int, float)):
        return [str(prediction)]
    elif isinstance(prediction, str):
        try:
            return json.loads(prediction) if prediction.strip().startswith("[") else [prediction]
        except:
            return [prediction]
    return []

def eval_rr(predictions, answer):
    """计算 Reciprocal Rank"""
    if answer is None or predictions is None:
        return 0.0
    for idx, pred in enumerate(predictions):
        if match(pred, answer):
            return 1.0 / (idx + 1)
    return 0.0


q2a = []

def get_fact(node):
    """安全获取事实数据，处理多级回退"""
    for key in ['facts1', 'facts2', 'facts']:
        if key in node and node[key] is not None:
            fact = node[key]
            # 处理图片中显示的字符串格式问题
            return fact
    return None

trees = json.load(open("../results/test_full.json", "r"))

acc_list = []
hit_list = []
f1_list = []
precission_list = []
recall_list = []
rr_list = []
hit_by_qlevel = defaultdict(lambda: {"hit": 0, "total": 0})
rr_by_qlevel = defaultdict(list)

print(len(trees))
for i, tree in enumerate(trees):
    node = tree[-1]
    question, prediction = node["question"], node["answer"][0]
    normalized_pred = normalize_prediction(prediction)
    topk_pred = topk(normalized_pred, 1)
    fact = get_fact(node)
    
    gold = node["gold_answer"]
    qlevel = node["qlevel"]
    hit = eval_hit(topk_pred[0], gold)  # top1 hit
    rr = eval_rr(topk_pred, gold) 
    #print(rr)      # MRR 用 topk_pred 全部计算
    rr_list.append(rr)
    rr_by_qlevel[qlevel].append(rr)  # 添加到对应分类
    if hit == 0:
        q2a.append({"question" : question, "gold_answer": gold, "prediction": prediction, "facts": fact})
    hit_list.append(hit)
    hit_by_qlevel[qlevel]["hit"] += hit
    hit_by_qlevel[qlevel]["total"] += 1
    
print(f"Overall Hit: {sum(hit_list) * 100 / len(hit_list):.2f}% ({sum(hit_list)}/{len(hit_list)})")
print(f"Overall MRR: {sum(rr_list) / len(rr_list):.4f}")
print("Hit by qlevel:")
for atype, stats in hit_by_qlevel.items():
    hit, total = stats["hit"], stats["total"]
    acc = hit * 100 / total if total > 0 else 0.0
    print(f"  {atype}: {acc:.2f}% ({hit}/{total})")

print("Hit and MRR by qlevel:")
for atype, stats in hit_by_qlevel.items():
    hit, total = stats["hit"], stats["total"]
    acc = hit * 100 / total if total > 0 else 0.0
    mrr = sum(rr_by_qlevel[atype]) / len(rr_by_qlevel[atype]) if rr_by_qlevel[atype] else 0.0
    print(f"  {atype}: Hit = {acc:.2f}% ({hit}/{total}), MRR = {mrr:.4f}")

json.dump(q2a, open("q2afact.json", "w"), indent=2)