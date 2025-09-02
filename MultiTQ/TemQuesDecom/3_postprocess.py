import json
from tqdm import tqdm
from termcolor import colored
import os

raw_data = json.load(open('outputs_100/predictions_100.json'))
total_items = len(raw_data)
processed_items = 0
error_items = 0
empty_responses = 0
data = {}
for item in tqdm(raw_data):
    processed_items += 1
    
    prompt = item['prompt']
    question = prompt.split('\n')[-2][len('Q: '):].strip()
    #print(colored(question, 'red'))
    try:
        hqdt = item['response']
    except:
        hqdt = None
    
    if hqdt is None:
        print(colored("Error: ", 'red'), "No decomposition found")
        data[question] = {}
        empty_responses +=1
        continue
    
    qds = {}
    for sub_question, qd in hqdt.items():
        
        qd_score = 0
        qds[sub_question] = (qd, qd_score)
        #print(colored(sub_question, 'blue'))
        #print(qd)
    
    
    data[question] = qds
json.dump(data, open('question_decompositions_100.json', 'w'), indent = 2)
# 打印统计信息
print(f"总条目数: {total_items}")
print(f"处理的问题数: {processed_items}")
print(f"输出的问题数: {len(data)}")
print(f"错误项数: {empty_responses}")