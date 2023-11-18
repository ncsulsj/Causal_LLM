"""
Script to test GPT-3.5 or GPT-4
"""

from causaleval import *
import time 
from dotenv import load_dotenv
import warnings
import argparse
warnings.filterwarnings("ignore")


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

### Add argument parser here is necessary (define the system prompt)


class GPT4(object):

    def __init__(self):
        pass

    def predict(self, prompt):

        response = openai.ChatCompletion.create(
                     model="gpt-4",
                    messages=[{"role": "system", "content":  """
        You are a helpful assistant to suggest potential causal pairs with direction (A -> B means A causes B)
        """},
                    {"role": "user", "content": prompt
                     }],
                     temperature=0,
                    )
        answer = response["choices"][0]["message"]["content"]
        return answer


def main():

    time1 = time.time()
    galton = pd.read_csv(
    "cmu.edu_dietrich_causality_assets_data_Galton_processed.txt", sep="\t")
    data = galton.drop(['family'], axis=1).reset_index(drop=True)
    data.columns = ["Father's Height",
                "Mother's Height", "Gender", "Child's Height"]
    relations = [("Father's Height", "Child's Height"), ("Mother's Height", "Child's Height"),
                 ("Gender", "Child's Height")]
    Llm = GPT4()

    eval = causal_eval(data, relations, Llm)

    print(eval.evaluate(10, 8000, 0.5))
    time2 = time.time()

    print(time2 - time1)


if __name__ == "__main__":
    
    main()



