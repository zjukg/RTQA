import json, jsonlines
import random
random.seed(42)

instruction_simple = '\n'.join([_.strip() for _ in open('./prompt/simple.txt').readlines()])
instruction_mc = '\n'.join([_.strip() for _ in open('./prompt/mc.txt').readlines()])
question_path = "../../datasets/TimelineKGQA/test_cron_questions.json"
with open(question_path, 'r') as file:
    question_json = json.load(file)
#question_list = [q['question'] for q in question_json]
prompts = []
sampled_questions = random.sample(question_json, min(1000, len(question_json)))
for q in sampled_questions:
    question = q['paraphrased_question']
    qlevel = q['question_level']
    if qlevel == 'simple':
        instruction = instruction_simple
    elif qlevel == 'medium' or qlevel == 'complex':
        instruction = instruction_mc
    
    prompt = instruction + '\nQ: ' + question + '\nA: '
    prompts.append(prompt)

json.dump(prompts, open('prompts_1000.json', 'w'), indent = 2)
print(len(prompts))
print(len(question_json))