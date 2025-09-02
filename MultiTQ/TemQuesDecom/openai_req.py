from openai import OpenAI
import requests
import time
import os
import json, jsonlines
import re

class OpenaiReq():
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.cache = {}
        self.cache_path = "./cache.jsonl"
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                for i, line in enumerate(f):
                    #print(i+1)
                    datum = json.loads(line.strip())
                    self.cache[tuple(datum["input"])] = datum["response"]
                f.close()

    
    
    def req2openai(self, prompt, model="gpt-4o-mini", temperature=0, max_tokens=128, stop=None, logprobs=True, use_cache=False):
        assert isinstance(prompt, str)
        input = (prompt, model, max_tokens, stop, logprobs)
        if use_cache and temperature == 0 and input in self.cache:
            return self.cache[input], True
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        for i in range(3):
            try:
                messages = [
                    {"role": "user", "content": prompt}
                ]
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                    logprobs=logprobs
                )
                break
            except Exception as e:
                err_msg = str(e)
                print(e)
                if "reduce your prompt" in err_msg: # this is because the input string too long
                    return ['too long'], False
        try:
            response = response.choices[0].message.content
        except:
            return ['openai error'], False
        if temperature == 0:
            input = (prompt, model, max_tokens, stop, logprobs)
            #res = response[0] if isinstance(response, str) else response 
            if input not in self.cache:
                self.cache[input] = [response]
                with open(self.cache_path, "a") as f:
                    f.write("%s\n"%json.dumps({"input": input, "response": [response]}))
                    f.close()
        return response, True

if __name__ == "__main__":
    api_key="xxx"
    base_url="https://"
    caller = OpenaiReq(api_key,base_url)
    res = caller.req2openai("你好", use_cache=True)
    print(res)
    
    