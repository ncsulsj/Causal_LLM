"""
Script to test the capability of LLMs for causal discovery 
"""

from src.causaleval import *
import time 
from dotenv import load_dotenv
import warnings
import argparse
import boto3
import json
from abc import ABC
import argparse
warnings.filterwarnings("ignore")
import ast 


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
brt = boto3.client(service_name='bedrock-runtime', region_name = "us-east-1")


### Add Logger here is necessary (define the system prompt)

class LLM(ABC): 

    def __init__(self):
        pass

    def predict(self, prompt):
        raise NotImplementedError
    

class GPT4(LLM):

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, prompt):

        response = openai.ChatCompletion.create(
                     model= self.model_name,
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

def parse_args():

    parser = argparse.ArgumentParser(description= "Evaluate the capability of LLMs on the causal discovery task")

    parser.add_argument("--llm", type=str, default="gpt4", help="The LLM to be evaluated")
    parser.add_argument("--data", type=str, default="data/sachs.txt", help="The path of the dataset to be used")
    parser.add_argument("--relations", type = str, default = "relation/relation.txt", help = "The true relations to be evalutaed")
    parser.add_argument("--task_type", type = str, default = "two_variable_discovery", help = "The type of task to be evaluated")
    parser.add_argument("--linear_coefficient", type = float, default = 1, help = "The linear coefficient in the simulation model of pairwise causal discovery")
    parser.add_argument("--df", type = int, default = 3, help = "The degree of freedom in the simulation model of pairwise causal discovery")
    parser.add_argument("--count", type = int, default = 2, help = "The number of trials to be evaluated")
    parser.add_argument("--max_tokens", type = int, default = 8000, help = "The maximum number of tokens to be used in the evaluation")
    parser.add_argument("--reserved_ratio", type = float, default = 0.5, help = "The ratio of reserved tokens in the evaluation")
    parser.add_argument("--model_name", type = str, default = "gpt-4-1106-preview", help = "The name of the LLM to be evaluated")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()
    start_time = time.time()
    print("Evaluation Begins ====================== >>>> ")
    try:
        eval_data = pd.read_csv(args.data, sep= "\t")
    except Exception as e:
        print("A data import error occured:", str(e))
    try:
        with open(args.relations, "r") as file:
            relations = ast.literal_eval(file.read())
    except Exception as e:
        print("A relation import error occured:", str(e)) 
    
    match args.llm:
        case "gpt4":
            Llm = GPT4(args.model_name)
        case "claude2":
            Llm = Claude2()
        case "jurassic":
            Llm = Jurassic()
        case "llama2":
            Llm = Llama2()
        case _:
            raise ValueError("The LLM is not supported")
        
    eval = causal_eval(eval_data, relations, Llm)

    if args.task_type == "full_graph_discovery":
        print(eval.evaluate(args.count, args.max_tokens, args.reserved_ratio))
    elif args.task_type == "two_variable_discovery":
        print(eval.two_variable_evaluate(linear_coefficient= args.linear_coefficient, df = args.df, count=args.count, max_tokens=args.max_tokens, reserved_ratio= args.reserved_ratio))
    else:
        raise ValueError("The task type is not supported")
    
    end_time = time.time()

    print("The evaluation time spends: " + str(end_time - start_time))


if __name__ == "__main__":
    
    main()



