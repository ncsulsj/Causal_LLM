# Is Knowledge all Large Language Models Needed for Causal Reasoning?

This includes the original implementation of **Is Knowledge all Large Language Models Needed for Causal Reasoning?**


In this implementation, we encapsulate the series of novel experiments that generate counterfactual settings to estimate attribution scores of LLMs in the proposed causal attribution model. We also include two additional experiments: reversal causal inference and pairwise causal discovery. In each case, true discovery rate, false discovery rate, F1 score, and Structual Hamming Distance are also provided to improve the understanding of LLMs' ability to do causal reasoning.

Our implementation can be easily adapted to testing other LLMs and settings by defining the interface of them without modifying the source code.


## Content 
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Input Argument](#input-argument)
4. [Run Example](#run-example)
5. [Output Explaination](#output-explaination)

## Installation
Install dependent Python libraries by running the command below.

```
pip install -r requirements.txt
```
Please use python version >= 3.10.0, as the **Match Case** is not applied into the older version. Besides, you need to create **.env** file to store the openai api key.

## Quick start
Once you finish the environment setup, run the command below to run the test script with the default input argument 
```
python test.py
```
The default setup tests the full graph causal discovery capability of GPT4 turbo on the Sachs dataset. 

## Input Argument 

- **llm**: The LLMs model to be tested. User could choose from [gpt4, claude2, jurassic, llama2]. 
- **data**: The path of the data to be tested. It needs to be the format of csv or txt file, separated by "\t".
- **relations**: The path of the true causal relations for the data to be tested. It needs to be the format of a txt file. For example, if the true causal relationships are Father's Height -> Child's Height, Mother's Height -> Child's Height, and Child's Gender -> Child's Height. It should store [("Father's Height", " Child's Height"), ("Mother's Height", "Child's Height"), ("Child's Gender", "Child's Height")] in the text file.
- **task_type**: There are two types of tasks: full_graph_discovery and two_variable_discovery.
- **linear_coefficient**: The linear coefficient in the simulation setting of pairwise causal discovery.
- **df**: The degree of freedom (parameter of Chi-square distribution) in the simulation setting of pairwise causal discovery.
- **count**: The number of trials to be tested
- **max_tokens**: The maximum number of tokens to be used in the evaluation (usually the context length of LLMs)
- **reserved_ratio**: The ratio of the max_tokens to be used in the evaluation test (to control the amount of numerical data used in the trial)
- **model_name**: The OpenAI GPT model to be tested. User could choose from [gpt-3.5-turbo, gpt-4, gpt-4-1106-preview]

## Run Example 

### Full Graph Discovery 

To test full causal graph discovery capability of GPT4 on the Sachs Data for 15 trials, run following command below 

```
python test.py --llm gpt4 \
               --data ./data/sachs.txt \
               --relations ./relation/relation.txt \
               --task_type full_graph_discovery \
               --count 15 \
               --max_tokens 8000 \
               --reserved_ratio 0.5 \
               --model_name gpt4
```
To test pairwise causal discovery capability of GPT4 on the Sachs Data for 15 trials, run following command below 

```
python test.py --llm gpt4 \
               --data ./data/sachs.txt \
               --relations ./relation/relation.txt \
               --task_type two_variable_discovery \
               --linear_coefficient 1 \
               --df 3 \
               --count 15 \
               --max_tokens 8000 \
               --reserved_ratio 0.5 \
               --model_name gpt4
```
## Output Explaination 

**Illustration of output for full causal graph discovery (2 trials)**

```
({1: {'tdr': (0.55, 0.04999999999999999), 'fdr': (0.45, 0.04999999999999999), 'F1': (0.5499950000458329, 0.049999999995833405), 'shd': (16.5, 0.5)}, 
  2: {'tdr': (0.72275, 0.17725000000000002), 'fdr': (0.27725, 0.17725000000000002), 'F1': (0.7227450000368034, 0.1772499999909743), 'shd': (16.0, 0.0)}, 
  3: {'tdr': (0.05, 0.05), 'fdr': (0.95, 0.04999999999999999), 'F1': (0.049997500124993745, 0.049997500124993745), 'shd': (29.5, 5.5)}, 
  4: {'tdr': (0.05, 0.05), 'fdr': (0.95, 0.04999999999999999), 'F1': (0.049997500124993745, 0.049997500124993745), 'shd': (21.5, 0.5)}, 
  5: {'tdr': (0.75, 0.25), 'fdr': (0.5, 0.0), 'F1': (0.5833286111509256, 0.0833336111009261), 'shd': (17.5, 1.5)}, 
  6: {'tdr': (0.0, 0.0), 'fdr': (1.0, 0.0), 'F1': (0.0, 0.0), 'shd': (20.0, 0.0)}}, 
  {'CAK': (0.5, 0.0), 'CAD': (-0.17275000000000001, 0.12725000000000003), 'MAD': (0.05, 0.05), 'MAK': (0.72275, 0.17725000000000002)})
```
- 1 denotes the Raw Data setting
- 2 denotes the Omit Data setting
- 3 denotes the Omit Knowledge setting
- 4 denotes the Reverse Causal Inference setting with reversed causal graph as the true causal graph
- 5 denotes the Reverse Causal Inference setting with original causal graph as the true causah graph
- 6 denotes the Random Guess setting
- tdr: true discovery rate, fdr: false discovery rate, F1: F1 score, shd: Structual Hamming Distance
- CAK: Conditional attribution of knowledge, CAD: Conditional attribution of data, MAK: Marginal attribution of Knowledge, MAD: Marginal attribution of data
- (*, *) denotes the mean and the standard deviation with respect to total number of trials

**Illustration of outout for pairwise causal graph discovery (2 trials)**

```
{'cdm': 1.0, 'llm': 0.5, 'llm_native': 0.0}
```
- cdm: mean accuracy of pairwise causal discovery by running LinGAM algorithm
- llm: mean accuracy of pairwise causal discovery by instructing LLM the procedure of LinGAM
- llm_native: mean accuracy of pairwise causal discovery by direclty asking LLM to generate the causal direction 
