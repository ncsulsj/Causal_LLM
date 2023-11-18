"""
Main function to evaluate the capability of LLMs to do causal discovery
"""

import networkx as nx 
import pandas as pd 
import tiktoken 
import numpy as np 
from random import shuffle
import openai 
import re 
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    ProcessPoolExecutor,
)
from typing import List, Tuple, Any
from copy import deepcopy
import os 

"""
word that does not exist in human language
"""

fake_words = ["bryoto", "nienet", "feouni", "alphan", "meedio", "enspop",
              "kivaro", "fuggen", "eosivo", "medent", "rademi", "rambli",
              "operia", "ativer", "kobans", "buffer", "ocoppu", "mezene",
              "sexuad", "protiv", "eyroag", "tilike", "gowwoo", "locarb"]



class causal_eval(object):

    def __init__(self, input_data: pd.DataFrame, relations: List[Tuple[str, str]], llm: Any):
        """
        :param input_data: data for the causal discovery 
        :param relations: correct directed relations. For example, [("A", "B"), ("C", "A")] states "A" -> "B" and "C" -> "A"
        :param llm: wrapper class around the LLMs such as GPT-4. should have class function ```predict``` to accept prompt and return the response
        """
        self.input_data = input_data
        self.relations = relations 
        self.llm = llm 
        self.system = """
        You are a helpful assistant to suggest potential causal pairs with direction (A -> B means A causes B)
        """
        self.user_prompt = """
        Suggest causal pairs with direction among following variables after analyzing following data::\n{data}. \n MUST Suggest 
        ONLY the directed causal pairs without saying any other things:
        """ 
        self.num_tokens, self.per_row_tokens = self.count_tokens()
        self.correct_relations = self.format_relations()
        self.reversed_relations, self.reversed_data = self.get_reversed_data()
        self.fake_relations, self.fake_data = self.fake_data_relation()

    def fake_data_relation(self):
        """
        create dataframe with columns as fake words and the corresponding correct directed relations
        """
        fa_map = {}
        cols = self.input_data.columns.to_list()
        for faw, rew in zip(fake_words[:len(cols)], cols):
            fa_map[rew] = faw
        fake_relations = [fa_map[relation[0]] + " -> " + fa_map[relation[1]]for relation in self.relations]
        fake_columns = [fa_map[name] for name in cols]
        fake_data = deepcopy(self.input_data)
        fake_data.columns = fake_columns

        return fake_relations, fake_data
    
    def get_reversed_data(self):
        """
        create dataframe corresponding to reversed topological order and get the reversed causal relations
        """
        graph = nx.DiGraph(self.relations)
        topo_order = list(nx.topological_sort(graph))
        reversed_topo_order = list(reversed(topo_order))
        adj_m = nx.adjacency_matrix(graph, nodelist = topo_order).todense()
        pairs_index = np.nonzero(adj_m)
        reversed_relations = [reversed_topo_order[pair[0]] + " -> " + reversed_topo_order[pair[1]] 
                              for pair in zip(*pairs_index)]
        reversed_data = self.input_data[topo_order]
        reversed_data.columns = reversed_topo_order

        return reversed_relations, reversed_data
    
    def count_tokens(self):
        """
        calculate the length of the prompt and the per row data length for truncation use
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(self.user_prompt.format(data = self.input_data.to_string())))
        per_row_tokens = len(encoding.encode(self.input_data.iloc[1].to_string()))

        return num_tokens, per_row_tokens

    def format_relations(self):
        """
        transfer ("A", "B") to "A" -> "B" 
        """

        return [relation[0] + " -> " + relation[1] for relation in self.relations]
    
    def truncate_data(self, max_tokens, reserved_ratio):
        """
        :param max_tokens: maximum of LLMs context length 
        :param reserved_ratio: the ratio of max_tokens to put the data into prompt
        """
        reserved_tokens = int(max_tokens * reserved_ratio)
        num_samples = int(reserved_tokens / self.per_row_tokens)
        total_samples = len(self.input_data)
        if num_samples < total_samples:
            random_indices = np.random.choice(total_samples, size = num_samples, replace = False)
            self.input_data = self.input_data.iloc[random_indices]
            self.reversed_data = self.reversed_data.iloc[random_indices]
            self.fake_data = self.fake_data.iloc[random_indices]

    def predict(self, prompt):
        """
        return the predicted causal pairs using provided LLMs
        """

        return self.llm.predict(prompt)
    
    def calculate_tdr_fdr(self, prompt, true_pairs):
        """
        return tdr and fdr through GPT-4
        """
        predicted_pairs = self.predict(prompt)
        response = openai.ChatCompletion.create(
                     model="gpt-4",
                    messages=[{"role": "system", "content": "You are a helpful assistant to calculate True discovery rate and False discovery rate (Ignore small spelling error and Letter case issue)"},
                    {"role": "user", "content": f"""
                     Cacluate the true discovery rate and false discovery rate: \n Correct pairs: {true_pairs} \n Predicted pairs: {predicted_pairs}
                    \n\n Return me with the floating point results true discovery rate and false discovery rate and don't give me the process of calculation.
                     Using the answer template: Tdr: Fdr: 
                     """
                     }],
                     temperature=0,
                    )
        answer = response["choices"][0]["message"]["content"]
        pattern = r"\d+\.\d+"
        match = re.findall(pattern, answer)
        if len(match) == 0:

            return 0, 1
        else:

            return float(match[0]), float(match[1])
    
    def evaluate_once(self, i):
        """
        :param i: for process indicator, not relevant with the code 

        get tdr and fdr for six scenairos described in our paper for one time: 
        (raw_data, perturb_data, perturb knowledge, reverse_causal_original, reverse_causal, and random guess)
        """
        cols = self.input_data.columns.to_list()
        shuffle(cols)
        cur_data = self.input_data[cols]
        cur_reversed_data = self.reversed_data[cols]
        fake_cols = self.fake_data.columns.to_list()
        shuffle(fake_cols)
        cur_fake_data = self.fake_data[fake_cols]

        # get prompts for six scenarios
        prompt1 = self.user_prompt.format(data = cur_data.to_string())
        prompt2 = self.user_prompt.format(data = cols)
        prompt3 = self.user_prompt.format(data = cur_fake_data.to_string())
        prompt4 = self.user_prompt.format(data = cur_reversed_data.to_string())
        prompt6 = self.user_prompt.format(data = fake_cols)

        with ThreadPoolExecutor(max_workers = 3) as executor:
            fu1 = executor.submit(self.calculate_tdr_fdr, prompt1, self.correct_relations)
            fu2 = executor.submit(self.calculate_tdr_fdr, prompt2, self.correct_relations)
            fu3 = executor.submit(self.calculate_tdr_fdr, prompt3, self.correct_relations)
            fu4 = executor.submit(self.calculate_tdr_fdr, prompt4, self.correct_relations)
            fu5 = executor.submit(self.calculate_tdr_fdr, prompt4, self.reversed_relations)
            fu6 = executor.submit(self.calculate_tdr_fdr, prompt6, self.fake_relations)  
        
            executor.shutdown()
             
            return {1: {"tdr": fu1.result()[0], "fdr": fu1.result()[1]}, 2: {"tdr": fu2.result()[0], "fdr": fu2.result()[1]}, 
                3: {"tdr": fu3.result()[0], "fdr": fu3.result()[1]}, 4: {"tdr": fu4.result()[0], "fdr": fu1.result()[1]},
                5: {"tdr": fu5.result()[0], "fdr": fu5.result()[1]}, 6: {"tdr": fu6.result()[0], "fdr": fu6.result()[1]}}    
        
    
    def evaluate(self, count, max_tokens, reserved_ratio = 0.8):
        """
        :param count: number of runs to evaluate LLMs
        :param max_tokens: maximum of LLMs context length 
        :param reserved_ratio: the ratio of max_tokens to put the data into prompt

        calculate the mean, std of tdr and fdr in each scenairo and provide the attribution scores
        """
        self.truncate_data(max_tokens, reserved_ratio)
        results = []
        with ProcessPoolExecutor(max_workers = 3 ) as executor:
            future_to_result = [executor.submit(self.evaluate_once,  i) for i in range(count)]
            for fut in as_completed(future_to_result):
                try:
                    result = fut.result()
                except Exception as exc:
                    print("Exception happens")
                else:
                    results.append(result)
        average_result = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
        for key in average_result.keys():
            average_result[key]["tdr"] = (np.mean([data[key]["tdr"] for data in results]), np.std([data[key]["tdr"] for data in results]))
            average_result[key]["fdr"] = (np.mean([data[key]["fdr"] for data in results]), np.std([data[key]["fdr"] for data in results]))
        CAK = average_result[1]["tdr"][0] - average_result[3]["tdr"][0]
        CAD = average_result[1]["tdr"][0] - average_result[2]["tdr"][0]
        MAD = average_result[3]["tdr"][0] - average_result[6]["tdr"][0]
        MAK = average_result[2]["tdr"][0] - average_result[6]["tdr"][0]

        return average_result, {"CAK": CAK, "CAD": CAD, "MAD": MAD, "MAK": MAK}