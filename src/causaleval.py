"""
Main function to evaluate the capability of LLMs to do causal discovery
"""

import networkx as nx 
import pandas as pd 
import tiktoken 
from src.utils import *
import numpy as np 
from random import shuffle
import openai 
import random 
import re 
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    ProcessPoolExecutor,
)
from typing import List, Tuple, Any
from copy import deepcopy
import os 
from causallearn.search.FCMBased import lingam

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
        self.nodes = set([node for relation in relations for node in relation])
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
        self.fake_nodes = set([self.fa_map[node] for node in self.nodes])

    def fake_data_relation(self):
        """
        create dataframe with columns as fake words and the corresponding correct directed relations
        """
        self.fa_map = {}
        cols = self.input_data.columns.to_list()
        for faw, rew in zip(fake_words[:len(cols)], cols):
            self.fa_map[rew] = faw
        fake_relations = [self.fa_map[relation[0]] + " -> " + self.fa_map[relation[1]]for relation in self.relations]
        self.fake_topo = [self.fa_map[node] for node in self.topo_order]
        fake_columns = [self.fa_map[name] for name in cols]
        fake_data = deepcopy(self.input_data)
        fake_data.columns = fake_columns

        return fake_relations, fake_data
    
    def get_reversed_data(self):
        """
        create dataframe corresponding to reversed topological order and get the reversed causal relations
        """
        graph = nx.DiGraph(self.relations)
        self.topo_order = list(nx.topological_sort(graph))
        reversed_topo_order = list(reversed(self.topo_order))
        self.adj_m = nx.adjacency_matrix(graph, nodelist = self.topo_order).todense()
        pairs_index = np.nonzero(self.adj_m)
        reversed_relations = [reversed_topo_order[pair[0]] + " -> " + reversed_topo_order[pair[1]] 
                              for pair in zip(*pairs_index)]
        reversed_data = self.input_data[self.topo_order]
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

    def _calculate_F1(self, tdr, fdr):
        """
        :param tdr: true discovery rate for one time run 
        :param fdr: false discovery rate for one time run

        calculate the f1 score 
        """  
        return calculate_F1(tdr, fdr)
    
    def _calculate_shd(self, correct, pred):
        """
        :param correct: true DAG 
        :param pred: predicted DAG

        calculate the structual hamming distance
        """
        return calculate_shd(correct, pred)
    
    def _calculate_tdr_fdr_shd(self, prompt, true_pairs, fake, reverse):
        """
        return tdr and fdr through GPT-4 turbo
        """
        predicted_pairs = self.predict(prompt)
        checked_pairs = check_error_pairs(format_string_to_list(predicted_pairs), self.fake_nodes if fake else self.nodes)
        if len(checked_pairs) == 0:
            pred_adj_m = np.zeros((len(self.topo_order), len(self.topo_order)))
            pred_adj_m_reverse = np.zeros((len(self.topo_order), len(self.topo_order)))
        else:
            predicted_nodes = set([node for pair in checked_pairs for node in pair])
            unpredicted_nodes = self.fake_nodes - predicted_nodes if fake else self.nodes - predicted_nodes
            checked_pairs.extend([(node, node) for node in unpredicted_nodes])
            predicted_graph = nx.DiGraph(checked_pairs)
            pred_adj_m = nx.adjacency_matrix(predicted_graph, nodelist = self.fake_topo if fake else self.topo_order).todense()
            pred_adj_m_reverse = nx.adjacency_matrix(predicted_graph, nodelist = list(reversed(self.topo_order))).todense() \
                                if reverse else np.zeros((len(self.topo_order), len(self.topo_order)))
            np.fill_diagonal(pred_adj_m, 0)
            np.fill_diagonal(pred_adj_m_reverse, 0)
        shd = self._calculate_shd(self.adj_m,   pred = pred_adj_m_reverse if reverse else pred_adj_m)
        return calculate_tdr_fdr(predicted_pairs, true_pairs) + (shd,)
    
    def two_variable_evalute_once(self, linear_coefficient, df, i):
        relation = random.choice(self.relations)
        result = {}
        dep, out = relation[1], relation[0]
        test_data = self.input_data[[dep, out]]
        test_data = add_noise_to_data(test_data, dep, out, linear_coefficient, df)
        cdm = lingam.ICALiNGAM(random_state = 42, max_iter = 100)
        cdm.fit(test_data)
        result["cdm"] = 1 if cdm.causal_order_ == [0, 1] else 0
        cols = test_data.columns.to_list()
        shuffle(cols)
        cur_data = test_data[cols]
        result["llm"], result["llm_native"] = two_variable_comparision(self.llm, cur_data, relation[1] + "->" + relation[0])
        return result 
 
    def two_variable_evaluate(self, linear_coefficient, df, count, max_tokens, reserved_ratio = 0.8):
        self.truncate_data(max_tokens, reserved_ratio)
        results = []
        with ProcessPoolExecutor(max_workers=3) as executor:
            future_to_result = [executor.submit(self.two_variable_evalute_once, linear_coefficient, df, i) for i in range(count)]
            for fut in as_completed(future_to_result):
                try:
                    result = fut.result()
                except Exception as exc:
                        print(exc)
                        print("Exception happens")
                else:
                    results.append(result)
        final_result = {"cdm":np.mean([result["cdm"]for result in results]), "llm":np.mean([result["llm"]for result in results]),
                        "llm_native": np.mean([result["llm_native"]for result in results])}
        return final_result

    
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
            fu1 = executor.submit(self._calculate_tdr_fdr_shd, prompt1, self.correct_relations, fake = False, reverse = False)
            fu2 = executor.submit(self._calculate_tdr_fdr_shd, prompt2, self.correct_relations, fake = False, reverse = False)
            fu3 = executor.submit(self._calculate_tdr_fdr_shd, prompt3, self.fake_relations, fake = True, reverse = False)
            fu4 = executor.submit(self._calculate_tdr_fdr_shd, prompt4, self.reversed_relations, fake = False, reverse = True)
            fu5 = executor.submit(self._calculate_tdr_fdr_shd, prompt4, self.correct_relations, fake = False, reverse = False)
            fu6 = executor.submit(self._calculate_tdr_fdr_shd, prompt6, self.fake_relations, fake = True, reverse = False)  
        
            executor.shutdown()
            return {1: {"tdr": fu1.result()[0], "fdr": fu1.result()[1], "shd": fu1.result()[2]}, 2: {"tdr": fu2.result()[0], "fdr": fu2.result()[1], "shd": fu2.result()[2]}, 
                3: {"tdr": fu3.result()[0], "fdr": fu3.result()[1], "shd": fu3.result()[2]}, 4: {"tdr": fu4.result()[0], "fdr": fu4.result()[1], "shd": fu4.result()[2]},
                5: {"tdr": fu5.result()[0], "fdr": fu5.result()[1], "shd": fu5.result()[2]}, 6: {"tdr": fu6.result()[0], "fdr": fu6.result()[1], "shd": fu6.result()[2]}}  
        
    
    def evaluate(self, count, max_tokens, reserved_ratio = 0.8):
        """
        :param count: number of runs to evaluate LLMs
        :param max_tokens: maximum of LLMs context length 
        :param reserved_ratio: the ratio of max_tokens to put the data into prompt

        calculate the mean, std of tdr and fdr in each scenairo and provide the attribution scores
        """
        self.truncate_data(max_tokens, reserved_ratio)
        results = []
        with ProcessPoolExecutor(max_workers = 3) as executor:
            future_to_result = [executor.submit(self.evaluate_once,  i) for i in range(count)]
            for fut in as_completed(future_to_result):
                try:
                    result = fut.result()
                except Exception as exc:
                    print(exc)
                    print("Exception happens")
                else:
                    results.append(result)
        average_result = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}}
        for key in average_result.keys():
            average_result[key]["tdr"] = (np.mean([data[key]["tdr"] for data in results]), np.std([data[key]["tdr"] for data in results]))
            average_result[key]["fdr"] = (np.mean([data[key]["fdr"] for data in results]), np.std([data[key]["fdr"] for data in results]))
            average_result[key]["F1"] = (np.mean([self._calculate_F1(data[key]["tdr"], data[key]["fdr"]) for data in results]),
                                          np.std([self._calculate_F1(data[key]["tdr"], data[key]["fdr"]) for data in results]))
            average_result[key]["shd"] = (np.mean([data[key]["shd"] for data in results]), np.std([data[key]["shd"] for data in results]))
        
        # CAK = average_result[1]["tdr"][0] - average_result[3]["tdr"][0]
        # CAD = average_result[1]["tdr"][0] - average_result[2]["tdr"][0]
        # MAD = average_result[3]["tdr"][0] - average_result[6]["tdr"][0]
        # MAK = average_result[2]["tdr"][0] - average_result[6]["tdr"][0]
        CAK_data = [ data[1]['tdr'] - data[3]['tdr'] for data in results]
        CAK = (np.mean(CAK_data), np.std(CAK_data))
        CAD_data = [ data[1]['tdr'] - data[2]['tdr'] for data in results]
        CAD = (np.mean(CAD_data), np.std(CAD_data))
        MAD_data = [ data[3]['tdr'] - data[6]['tdr'] for data in results]
        MAD = (np.mean(MAD_data), np.std(MAD_data))
        MAK_data = [ data[2]['tdr'] - data[6]['tdr'] for data in results]
        MAK = (np.mean(MAK_data), np.std(MAK_data))    

        return average_result, {"CAK": CAK, "CAD": CAD, "MAD": MAD, "MAK": MAK}
    
