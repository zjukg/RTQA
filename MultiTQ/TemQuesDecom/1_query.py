import json
import re
from openai_req import OpenaiReq
import random
from tqdm import tqdm
import os
from multiprocessing import Pool
from termcolor import colored
import atexit
import multiprocessing
random.seed(42)

MAX_SPLIT = min(64, os.cpu_count() or 8)
STEP = 4
def cleanup():
    """终止所有残留子进程"""
    for process in multiprocessing.active_children():
        process.terminate()

def clean_response(response_text):
    try:
        # 去掉markdown的```json 和 ```，并strip空格
        clean_text = re.sub(r"^```json|```$", "", response_text.strip(), flags=re.MULTILINE).strip()
        # 变成真正的字典
        data = json.loads(clean_text)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"原始文本: {response_text}")
        return {"error": "JSON解析失败"}

def query(rank, prompts,api_key, base_url):
    print('Process rank {} PID {} begin...'.format(rank, os.getpid()))
    reqor = OpenaiReq(api_key, base_url)
    start = int(len(prompts) * rank / MAX_SPLIT)
    end = int(len(prompts) * (rank + 1) / MAX_SPLIT)
    queries = prompts[start:end]
    #queries = prompts[int(len(prompts) * rank / MAX_SPLIT) : int(len(prompts) * (rank + 1) / MAX_SPLIT)]
    try:
        with open(f'outputs_full/rank_{rank}.json', 'w') as fout:
            if rank == 0:
                bar = tqdm(range((len(queries) + STEP - 1) // STEP))
            else:
                bar = range((len(queries) + STEP - 1) // STEP)
            for idx in bar:
                inputs = queries[idx * STEP : (idx + 1) * STEP]
                if len(inputs) == 0:
                    break
                gpt_results = []
                for prompt in inputs:
                    try:
                        result, tag = reqor.req2openai(prompt, max_tokens=512, stop='\n\n')
                        if tag:
                            result = clean_response(result)
                        gpt_results.append(result)
                    except Exception as e:
                        print(f"请求错误: {e}")
                        gpt_results.append({"error": str(e)})

                for prompt, res in zip(inputs, gpt_results):
                    print(f"Rank {rank}, writing result: {res}")
                    output_item = json.dumps({'prompt': prompt, 'response': res}, ensure_ascii=False)
                    fout.write(output_item + '\n')
                    fout.flush()
            #fout.close()
    except Exception as err:
        print(Exception, err)

if __name__=='__main__':
    atexit.register(cleanup)
    prompts = json.load(open('prompts_full.json'))
    output_dir = 'outputs_full'
    os.makedirs(output_dir, exist_ok=True)
    print("number of prompts: {}".format(len(prompts)))
    print('Parent process %s.' % os.getpid())
    with Pool(MAX_SPLIT) as pool:
        args = [(i, prompts, "xxx", "https://") 
                for i in range(MAX_SPLIT)]
        pool.starmap(query, args)
    total_items = 0
    for i in range(MAX_SPLIT):
        filename = os.path.join(output_dir, f'rank_{i}.json')
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                lines = f.readlines()
                items = len(lines)
                total_items += items
                print(f"File {filename}: {items} items")
        else:
            print(f"File {filename} does not exist!")
            
    print(f'All subprocesses done. Total processed items: {total_items}')