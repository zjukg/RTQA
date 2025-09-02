import multiprocessing as mp
mp.set_start_method("spawn", force=True) 
import json
import re
import os
from question_answering import *
from tqdm import tqdm
#from parallel import parallel_process_data
from multiprocessing import Pool
from functools import partial
import concurrent.futures
import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio

PROC_NUM = 1
cnt = 0
triple_list = []
triple_path="../../datasets/MultiTQ/kg/full.txt"
with open(triple_path, 'r', encoding='utf-8') as file:
    for line in file:
        triplets = line.strip().replace("_", " ").split('\t')
        triple_list.append(triplets)
semaphore = None 

async def limited_solve(tree, idx, retriever):
    async with semaphore:
        return await solve(tree, idx, retriever)
async def solve(tree, idx, retriever):
    #print(1)
    #gpu_id=idx%3
    #await retriever.load()
    global cnt
    cnt += 1
    print(cnt)
    try:
        for node in tree:
            question = node["question_text"].strip()
            ref_tokens = re.findall(r"#\d+", question)
            topic_entities = []
            relevant_facts= []
            for ref_token in ref_tokens:
                if "fa" in node and int(ref_token[1:]) <= len(tree[node["fa"]]["sons"]):
                    ref_idx = tree[node["fa"]]["sons"][int(ref_token[1:])-1]
                    if "answer" in tree[ref_idx]:
                        question = question.replace(ref_token, tree[ref_idx]["answer"][0])
                        topic_entities.append(tree[ref_idx]["answer"][0])
            node["question"] = question
            #node["LLM_answer"] = await get_LLM_answer(question)
            if ref_tokens == [] and node["idx"] > 0 and "qlabel" not in node:
                relevant_facts = tree[node["idx"]-1]["answer"]
                node["IR_answer"] = await get_IR_answer2(question, relevant_facts[0])
                node["facts2"] = relevant_facts[0]
                node["answer"] = node["IR_answer"]
            else:

                node["LLM_answer"] = await get_LLM_answer(question)
                node["IR_answer"], node["facts1"] = await get_IR_answer1(question, retriever)
                print(question)
                #print(node["facts"])
                if len(node["sons"]) > 0:
                    node["child_answer"] = tree[node["idx"]-1]["answer"]
                    node["answer"] = node["child_answer"]
                else:
                    node["answer"] = await aggregate_answer1(node)
                    #node["answer"] = node["LLM_answer"]
    except Exception as e:
        print("ERROR CASE")
        print(tree[-1])
        raise e
    return tree
    
async def main():
    global semaphore
    # Initialize the semaphore within the main event loop
    semaphore = asyncio.Semaphore(20)
    trees = json.load(open("trees_test4.json", "r"))
    print("Total:", len(trees), "| Start Processing...")
    retrievers = []
    for id in range(1):
        gpu_id = id % 3
        retriever = Retrieval_BGE('time','BAAI/bge-m3', triple_list, gpu_id=gpu_id)
        await retriever.load()
        retrievers.append(retriever)
    # 创建并发任务
    tasks = []
    for i, tree in enumerate(trees):
        task = asyncio.create_task(limited_solve(tree, i, retrievers[i % 1]))
        tasks.append(task)
    results = []
    for task in tqdm_asyncio(asyncio.as_completed(tasks), total=len(tasks)):
        result = await task
        results.append(result)
    # 保存结果
    os.makedirs("../results", exist_ok=True)
    with open("../results/test_xronlyLLMv3.json", "w") as f:
        json.dump(results, f, indent=2)    

if __name__ == "__main__":
    asyncio.run(main())
    