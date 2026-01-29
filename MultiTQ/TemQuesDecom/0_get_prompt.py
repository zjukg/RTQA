import json, jsonlines

instruction_after_first = '\n'.join([_.strip() for _ in open('./prompt/after_first.txt').readlines()])
instruction_before_after = '\n'.join([_.strip() for _ in open('./prompt/before_after.txt').readlines()])
instruction_before_last = '\n'.join([_.strip() for _ in open('./prompt/before_last.txt').readlines()])
instruction_equal_multi = '\n'.join([_.strip() for _ in open('./prompt/equal_multi.txt').readlines()])
instruction_equal = '\n'.join([_.strip() for _ in open('./prompt/equal.txt').readlines()])
instruction_first_last = '\n'.join([_.strip() for _ in open('./prompt/first_last.txt').readlines()])

question_path = "../../datasets/MultiTQ/questions/test.json"
#raw_data = jsonlines.open("../../datasets/MultiTQ/questions/test_500.json", "r")
with open(question_path, 'r') as file:
    question_json = json.load(file)
#question_list = [q['question'] for q in question_json]
prompts = []

for q in question_json:
    question = q['question']
    qtype = q['qtype']
    if qtype == "equal":
        instruction = instruction_equal
    elif qtype == "before_after":
        instruction = instruction_before_after
    elif qtype == "first_last":
        instruction = instruction_first_last
    elif qtype == "equal_multi":
        instruction = instruction_equal_multi
    elif qtype == "after_first":
        instruction = instruction_after_first
    elif qtype == "before_last":
        instruction = instruction_before_last
    prompt = instruction + '\nQ: ' + question + '\nA: '
    prompts.append(prompt)
"""
#消融实验，去掉问题分解
for q in question_json:
    question = q['question']
    instruction=instruction_equal
    prompt = instruction + '\nQ: ' + question + '\nA: '
    prompts.append(prompt)
"""
json.dump(prompts, open('prompts_full.json', 'w'), indent = 2)
print(len(prompts))
print(len(question_json))