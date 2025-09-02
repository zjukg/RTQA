import json
from tqdm import tqdm
from termcolor import colored
import os
def to_hqdt(question: str) -> dict:
    """
    将输入问题包装为 HQDT 格式，子问题为空。
    """
    
    return {question: []}
question_path = "../../datasets/TimelineKGQA/test_cron_questions.json"
with open(question_path, 'r') as file:
    question_json = json.load(file)


data = {}
for item in tqdm(question_json):
   
    question = item['paraphrased_question']
    hqdt = to_hqdt(question)
    
    
    qds = {}
    for sub_question, qd in hqdt.items():
        
        qd_score = 0
        qds[sub_question] = (qd, qd_score)
        #print(colored(sub_question, 'blue'))
        #print(qd)
    
    
    data[question] = qds
try:
    with open('question_decompositions_fullxr1.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(colored(f"\n✅ 成功保存到: {'question_decompositions_fullxr1.json'}", 'green'))
except Exception as e:
    print(colored(f"\n❌ 保存失败: {str(e)}", 'red'))
# 打印统计信息


print(f"输出的问题数: {len(data)}")

