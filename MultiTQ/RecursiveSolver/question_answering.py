#from openai_req import OpenaiReq
from openai_req2 import AsyncOpenaiReq
import asyncio
import aiohttp
import requests
import os
import csv
import pickle
import time
import json
import openai
from tqdm import tqdm
import argparse
from retriever_BGE import Retrieval_BGE
#from retriever_BM25 import HybridRetrieval
#from retriever_milvusBGE import Retrieval_milvusBGE
#from retrival_BERT import Retrieval_BERT
import openai
import numpy as np
import faiss
import asyncio
import re
from typing import Tuple


print(openai.__version__)
print(openai.__file__)
openai_caller = AsyncOpenaiReq("xxx", "https://")
#deepseek
#openai_caller = AsyncOpenaiReq("xxx", "https://")

def postprocessr1(response):
    if not response or not isinstance(response, list):
        return 'ERROR: invalid response format', -100, ""
    response = response[0]
    if response == 'too long':
        return 'ERROR: prompt too long', -100, ""
    if response == 'openai error':
        return 'ERROR: openai error', -100, ""
    if response['finish_reason'] != 'stop':
        return 'ERROR: finish reason "{}"'.format(response['finish_reason']), -100, ""
    
    cot = response.get('content', '').strip()
    if not cot:
        return 'ERROR: empty output', 0, ""
    answer_patterns = [
        r'the answer is:\s*([^\n\.]+)',  # "the answer is: Paris"
        r'the answer is\s*([^\n\.]+)'  # "the answer is Paris"
    ]
    
    answer = ""
    for pattern in answer_patterns:
        match = re.search(pattern, cot, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            break
    
    return answer, 0, cot

def postprocess(response):
    if not response or not isinstance(response, list):
        return 'ERROR: invalid response format', -100, ""
    response = response[0]
    if response == 'too long':
        return 'ERROR: prompt too long', -100, ""
    if response == 'openai error':
        return 'ERROR: openai error', -100, ""
    if response['finish_reason'] != 'stop':
        return 'ERROR: finish reason "{}"'.format(response['finish_reason']), -100, ""
    
    logprobs = response.get('logprobs', {})
    cot = response.get('content', '').strip()
    tokens = logprobs.get('tokens', [])
    token_logprobs = logprobs.get('token_logprobs', [])
    if len(token_logprobs) == 0:
        return 'ERROR: empty output', -100, cot
    pos = -1
    for idx, token in enumerate(tokens):
        #and idx + 3 < len(tokens) and tokens[idx + 3].strip() == ':'
        if token.strip() == 'the' and idx +1 <len(tokens) and tokens[idx+1].strip() == 'answer' and idx + 2 < len(tokens) and tokens[idx + 2].strip() == 'is' :
            pos = idx
            break
    if pos == -1:
        answer = ''
        answer_logprobs = []
    else:
        # 判断第 pos+3 个 token 是否为冒号（存在时跳过）
        if tokens[-1] == '.':
            if idx + 3 < len(tokens) and tokens[idx + 3].strip() == ':':
                answer_logprobs = token_logprobs[pos+4:-1]
                answer = cot.split('the answer is: ')[-1][:-1]
            else:
                answer_logprobs = token_logprobs[pos+3:-1]
                answer = cot.split('the answer is ')[-1][:-1]
        else:
            if idx + 3 < len(tokens) and tokens[idx + 3].strip() == ':':
                answer_logprobs = token_logprobs[pos+4:]
                answer = cot.split('the answer is: ')[-1]
            else:
                answer_logprobs = token_logprobs[pos+3:]
                answer = cot.split('the answer is ')[-1]
    cot_process = cot.split('the answer is ')[0].strip()
    cot_process_logprobs = token_logprobs[:pos]
    if len(cot_process_logprobs) == 0:
        cot_process_logprob = -100
    else:
        cot_process_logprob = sum(cot_process_logprobs) / len(cot_process_logprobs)
    return answer, cot_process_logprob, cot

async def get_LLM_answer(question):
    instruction = '\n'.join([_.strip() for _ in open('LLM/prompt.txt').readlines()])
    prompt = instruction + '\nQ: ' + question + '\nA:'
    response, tag = await openai_caller.req2openai(prompt=prompt, temperature=0.7, max_tokens=1024, use_cache=False)
    return postprocess(response)

async def get_IR_answer1(question, retriever):
    distances, corpus_ids = await retriever.compute_similarity(question, n=50)
    fact_list = await retriever.get_result(question, distances, corpus_ids, re_rank=False)
    
    #dense_results, sparse_results, hybrid_results = await retriever.get_result(question, limit=40)
    # generate prompt
    fact = fact_list['fact']
    #fact = hybrid_results
    instruction = '\n'.join([_.strip() for _ in open('IR/prompt.txt').readlines()])
    if fact:
        fact_text = '\n'.join(fact)
    else:
        fact_text = 'No facts provided.'
    prompt = instruction + '\nHistorical facts:\n' + fact_text
    prompt += '\nQuestion: ' + question + '\nAnswer:'
    response, tag = await openai_caller.req2openai(prompt=prompt, temperature=0, max_tokens=1024, use_cache=False)
    return postprocess(response), fact_text

async def get_IR_answer2(question, relevant_facts):
    # generate prompt
    fact = relevant_facts
    #fact = hybrid_results
    instruction = '\n'.join([_.strip() for _ in open('IR/prompt.txt').readlines()])
    if fact:
        fact_text = '\n'.join(fact)
    else:
        fact_text = 'No facts provided.'
    prompt = instruction + '\nRelevant facts:\n' + fact_text
    prompt += '\nQuestion: ' + question + '\nAnswer:'
    response, tag = await openai_caller.req2openai(prompt=prompt, temperature=0, max_tokens=1024, use_cache=False)
    return postprocess(response)

def get_embdding_answer(node):
    ans=1
    return ans

async def aggregate_answer1(node):
    question = node["question"]
    instruction = '\n'.join([_.strip() for _ in open('aggregate/prompt.txt').readlines()])
    LLM_ans, LLM_score, LLM_cot = node["LLM_answer"]
    IR_ans, IR_score, IR_cot = node["IR_answer"]
    prompt = instruction + '\nQuestion:' + question + '\nCandidate answers:\nsource A: ' + LLM_ans + '\nsource B: ' + IR_ans
    if "child_answer" in node:
        child_ans, child_score, child_cot = node["child_answer"]
        prompt += '\nsource C: ' + child_ans
    prompt += '\nOutput: '
    response, tag = await openai_caller.req2openai(prompt=prompt, temperature=0, max_tokens=1024, use_cache=False)
    if tag == False:
        if "ERROR" in LLM_ans or 'Unknown' in LLM_ans:
            LLM_ans, LLM_score = "", -100
        if "ERROR" in IR_ans or 'Unknown' in IR_ans:
            IR_ans, IR_score = "", -100
        if "child_answer" not in node:
            res = max([(LLM_ans, LLM_score, LLM_cot), (IR_ans, IR_score, IR_cot)], key=lambda x:x[1])
        else:
            if "ERROR" in child_ans or 'Unknown' in child_ans:
                child_ans, child_score = "", -100
            res = max([(LLM_ans, LLM_score, LLM_cot), (IR_ans, IR_score, IR_cot), (child_ans, child_score, child_cot)], key=lambda x:x[1])
        return res
    return postprocess(response)

def aggregate_answer(node):
    LLM_answer = node["LLM_answer"]
    IR_answer = node["IR_answer"]
    LLM_ans, LLM_score, LLM_cot = LLM_answer
    IR_ans, IR_score, IR_cot = IR_answer
    if "ERROR" in LLM_ans or 'Unknown' in LLM_ans:
        LLM_ans, LLM_score = "", -100
    if "ERROR" in IR_ans or 'Unknown' in IR_ans:
        IR_ans, IR_score = "", -100
    res = max([(LLM_ans, LLM_score, LLM_cot), (IR_ans, IR_score, IR_cot)], key=lambda x:x[1])
    return res
    
def calculate_score1(cot_process_logprob, qd_score, sub_answer_scores):
    return cot_process_logprob + qd_score + sum(sub_answer_scores)

def calculate_score2(cot_process_logprob, qd_score, sub_answer_scores):
    return (cot_process_logprob + qd_score + sum(sub_answer_scores)) / (len(sub_answer_scores) + 2)

async def get_child_answer(node, tree):
    instruction = '\n'.join([_.strip() for _ in open('child/prompt.txt').readlines()])
    question = node["question"]
    qd_score = node["qd_logprob"]
    context = ''
    sub_answer_scores = []
    for son_idx in node["sons"]:
        sub_question = tree[son_idx]["question"]
        sub_answer = tree[son_idx]["answer"][0]
        sub_answer_scores.append(tree[son_idx]["answer"][1])
        context += sub_question + ' -- ' + sub_answer + '\n'
    prompt = instruction + '\nQuestion:\n{}\nContext:\n{}Answer:'.format(question, context)
    #prompt = instruction + '\nContext:\n{}\n\nQuestion:\n{}\n\nAnswer:'.format(context, question)
    response, tag = await openai_caller.req2openai(prompt=prompt, temperature=0, max_tokens=1024, use_cache=False)
    child_answer, cot_process_logprob, child_cot = postprocess(response)
    child_ans = child_answer
    child_score = calculate_score2(cot_process_logprob, qd_score, sub_answer_scores)
    res = (child_ans, child_score, child_cot)
    return res

async def main():
    question = "Who was the first country that Ethiopia expressed optimism about?"
    #r = bm25_search(question, k=5)
    triple_list = []
    triple_path="../../datasets/MultiTQ/kg/full.txt"
    with open(triple_path, 'r', encoding='utf-8') as file:
        for line in file:
            triplets = line.strip().replace("_", " ").split('\t')
            triple_list.append(triplets)
    retriever = Retrieval_BGE('time','BAAI/bge-m3', triple_list, gpu_id=0)
    await retriever.load()
    IR_ans , fact_text = await get_IR_answer1(question, retriever)
    print(fact_text)

if __name__ == "__main__":
    asyncio.run(main())
    #LLM_ans = asyncio.run(get_LLM_answer(question))
    #print(LLM_ans)
    
    
    

