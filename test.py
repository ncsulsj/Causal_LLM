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
                     model="gpt-3.5-turbo",
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
    d1 = pd.read_csv("pair0022.txt", sep=" ", names = ["Age", 'Height'])
    d2 = pd.read_csv("pair0023.txt", sep=" ", names = ["Age", 'Weight']).drop(["Age"], axis = 1)
    d3 = pd.read_csv("pair0024.txt", sep=" ", names = ["Age", 'Heart rate']).drop(["Age"], axis = 1)
    dwd = pd.concat([d1, d2, d3], axis = 1).dropna()


    relations = [("Age", "Height"), ("Age", "Weight"), ("Age", "Heart rate")]
    
    Llm = GPT4()

    eval = causal_eval(dwd, relations, Llm)

    print(eval.evaluate(15, 8000, 0.5))
    time2 = time.time()

    print(time2 - time1)


if __name__ == "__main__":
    
    main()



