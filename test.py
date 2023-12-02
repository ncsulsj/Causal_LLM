"""
Script to test GPT-3.5 or GPT-4
"""

from causaleval import *
import time 
from dotenv import load_dotenv
import warnings
import argparse
import boto3
import json
from abc import ABC
warnings.filterwarnings("ignore")


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
brt = boto3.client(service_name='bedrock-runtime', region_name = "us-east-1")


### Add argument parser here is necessary (define the system prompt)

class LLM(ABC): 

    def __init__(self):
        pass

    def predict(self, prompt):
        raise NotImplementedError
    

class GPT4(LLM):

    def __init__(self):
        pass

    def predict(self, prompt):

        response = openai.ChatCompletion.create(
                     model="gpt-4-1106-preview",
                    messages=[{"role": "system", "content":  """
        You are a helpful assistant to suggest potential causal pairs with direction (A -> B means A causes B)
        """},
                    {"role": "user", "content": prompt
                     }],
                     temperature=0,
                    )
        answer = response["choices"][0]["message"]["content"]
        return answer

class Claude2(LLM):

    def __init__(self):
        pass
    
    def predict(self, prompt):
        body = json.dumps({
        "prompt": "\n\nHuman: {}\n\nAssistant:".format(prompt),
        "max_tokens_to_sample": 300,
        "temperature": 0,
        "top_p": 0.9,
        })
        modelId = 'anthropic.claude-v2'
        accept = 'application/json'
        contentType = 'application/json'
        response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        return response_body.get('completion')
    

class Jurassic(LLM):

    def __init__(self):
        pass

    def predict(self, prompt):
        
        body = json.dumps({
        "prompt": "\n\nHuman: {}\n\nAssistant:".format(prompt),
        "maxTokens": 300,
        "temperature": 0,
        "topP": 0.9
        })
        modelId = 'ai21.j2-ultra-v1'
        accept = 'application/json'
        contentType = 'application/json'
        response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        response_body = json.loads(response.get('body').read())
        
        return response_body.get('completions')[0].get('data').get('text')
    

class Llama2(LLM):

    def __init__(self):
        pass 

    def predict(self, prompt):
        body = json.dumps({
        "prompt":  "\n\nHuman: {}\n\nAssistant:".format(prompt),
        "max_gen_len": 300,
        "temperature": 0,
        "top_p": 0.9,
        })
        modelId = 'meta.llama2-13b-chat-v1'
        accept = 'application/json'
        contentType = 'application/json'
        response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
        return json.loads(response.get('body').read())['generation']



def main():

    time1 = time.time()
    sachs = pd.read_csv("sachs.txt", sep="\t")


    relations = [("erk", "akt"), ("mek", "erk"), ("pip2", "pkc"),
                 ("pip3", "akt"), ("pip3", "pip2"), ("pip3", "plc"),
                 ("pka", "akt"), ("pka", "erk"), ("pka", "jnk"), ("pka", "mek"),
                 ("pka", "p38"), ("pka", "raf"), ("pkc", "jnk"), ("pkc", "mek"),
                 ("pkc", "p38"), ("pkc", "pka"), ("pkc", "raf"), ("plc", "pip2"),
                 ("plc", "pkc"), ("raf", "mek")]
    
    Llm = GPT4()

    eval = causal_eval(sachs, relations, Llm)

    print(eval.two_variable_evaluate(linear_coefficient= 2, df = 3, count=20, max_tokens=8000, reserved_ratio= 0.5))
    # print(eval.evaluate(15, 8000, 0.5))
    time2 = time.time()

    print(time2 - time1)


if __name__ == "__main__":
    
    main()



