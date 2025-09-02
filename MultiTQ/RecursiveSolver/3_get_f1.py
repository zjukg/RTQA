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
    try:
        if isinstance(answer, str):
            answer = ast.literal_eval(answer)
    except Exception:
        # 如果转换失败，说明 answer 本来就不是 list，而是一个普通字符串
        answer = [answer]
    #print(f"Raw answer: {answer}")
    for a in answer:
        if match(prediction, a):
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


q2a = []
q2a_trees = []

trees = json.load(open("../results/llama27bfine100.json", "r"))

acc_list = []
hit_list = []
f1_list = []
precission_list = []
recall_list = []
hit_by_answer_type = defaultdict(lambda: {"hit": 0, "total": 0})
hit_by_qlabel = defaultdict(lambda: {"hit": 0, "total": 0})
hit_by_equal = defaultdict(lambda: {"hit": 0, "total": 0})
hit_by_before_after = defaultdict(lambda: {"hit": 0, "total": 0})
hit_by_equal_multi = defaultdict(lambda: {"hit": 0, "total": 0})
print(len(trees))
for i, tree in enumerate(trees):
    node = tree[-1]
    question, prediction = node["question"], node["answer"]
    normalized_pred = normalize_prediction(prediction)
    topk_pred = topk(normalized_pred, 1)
    
    gold = node["gold_answer"]
    qlabel = node["qlabel"]
    qtype = node["qtype"]
    answer_type = node["answer_type"]
    time_level = node["time_level"]
    hit = eval_hit(topk_pred, gold)
    hit_list.append(hit)
    if(hit==0):
        q2a_trees.append(
            tree
        )
        q2a.append({"question": question, "prediction": prediction, "gold_answer": gold, "qlabel": qlabel, "qtype": qtype, "answer_type": answer_type, "time_level": time_level})
    hit_by_answer_type[answer_type]["hit"] += hit
    hit_by_answer_type[answer_type]["total"] += 1
    hit_by_qlabel[qlabel]["hit"] += hit
    hit_by_qlabel[qlabel]["total"] += 1
    if qtype == "equal":
        hit_by_equal[time_level]["hit"] += hit
        hit_by_equal[time_level]["total"] += 1
    elif qtype == "before_after":
        hit_by_before_after[time_level]["hit"] += hit
        hit_by_before_after[time_level]["total"] += 1
    elif qtype == "equal_multi":
        hit_by_equal_multi[time_level]["hit"] += hit
        hit_by_equal_multi[time_level]["total"] += 1
print(f"Overall Hit: {sum(hit_list) * 100 / len(hit_list):.2f}% ({sum(hit_list)}/{len(hit_list)})")
print("Hit by Answer Type:")
for atype, stats in hit_by_answer_type.items():
    hit, total = stats["hit"], stats["total"]
    acc = hit * 100 / total if total > 0 else 0.0
    print(f"  {atype}: {acc:.2f}% ({hit}/{total})")

# 输出按 qlabel 分类的命中率
print("Hit by QLabel:")
for qlabel, stats in hit_by_qlabel.items():
    hit, total = stats["hit"], stats["total"]
    acc = hit * 100 / total if total > 0 else 0.0
    print(f"  {qlabel}: {acc:.2f}% ({hit}/{total})")

print("Hit by Equal:")
for qlabel, stats in hit_by_equal.items():
    hit, total = stats["hit"], stats["total"]
    acc = hit * 100 / total if total > 0 else 0.0
    print(f"  {qlabel}: {acc:.2f}% ({hit}/{total})")

print("Hit by Before_after:")
for qlabel, stats in hit_by_before_after.items():
    hit, total = stats["hit"], stats["total"]
    acc = hit * 100 / total if total > 0 else 0.0
    print(f"  {qlabel}: {acc:.2f}% ({hit}/{total})")

print("Hit by Equal_Multi:")
for qlabel, stats in hit_by_equal_multi.items():
    hit, total = stats["hit"], stats["total"]
    acc = hit * 100 / total if total > 0 else 0.0
    print(f"  {qlabel}: {acc:.2f}% ({hit}/{total})")
   
json.dump(q2a, open("q2a.json", "w"), indent=2)
output_path = "q2a_full_tree.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(q2a_trees, f, indent=2, ensure_ascii=False)