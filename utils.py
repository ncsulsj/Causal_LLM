import numpy as np 
import openai 
import re 
from ast import literal_eval as eval 

def calculate_F1(tdr, fdr):
    """
    :param tdr: true discovery rate for one time run 
    :param fdr: false discovery rate for one time run

    calculate the f1 score 
    """  
    f1 = 2 * (1 - fdr) * tdr / (1 - fdr + tdr + 1e-5)
    return f1


def calculate_shd(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return shd


def calculate_tdr_fdr(predicted_pairs, true_pairs):
    """
    :param predicted_pairs: prediction output from the LLMs 
    :param true_pairs: true causal pairs

    return tdr and fdr through GPT-4 turbo
    """
    response = openai.ChatCompletion.create(
                     model="gpt-4-1106-preview",
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


def format_string_to_list(predicted_pairs):

    response = openai.ChatCompletion.create(
                     model="gpt-4-1106-preview",
                    messages=[{"role": "system", "content": "You are a helpful assistant to transform the predicted causal pairs to the following provided template"},
                    {"role": "user", "content": f"""
                     -------------------------------------------------------------------------------------------------
                     Here is an example to show how to transform: \n\n
                     predicted pairs: A->B. D->C, M->Q   after transformation: "[('A', 'B'), ('D', 'C'), ('M', 'Q')]"
                    -------------------------------------------------------------------------------------------------
                     The predicted pairs need to be transformed is following (you may need to clean it): \{[predicted_pairs]}\ \n
                     only give me the answer without stating the process and the instruction
                     """
                     }],
                     temperature=0,
                    )['choices'][0]['message']['content']
    
    try:
        while not isinstance(response, list):
            # need to add control to avoid dead loop
            response = eval(response)
    except:
        return []

    return response 


def check_error_pairs(predicted_pairs, nodes):

    pairs = []
    for pair in predicted_pairs:
        if pair[0] in nodes and pair[1] in nodes:
            pairs.append(pair)
    
    return pairs 


def add_noise_to_data(data, dep, outcome, linear_coefficient, df):

    data[outcome] = linear_coefficient * data[dep] + np.random.chisquare(df, size = (len(data),))

    return data 



def two_variable_comparision(llm, data, true):

    prompt1 = """
    Here is how LinGAM works to derive the causal pairs: 

    Given the data with columns [A, B]: \n
    
    1. You first fit the linear regression with column A as the feature and B as the outcome variable. Collect the fitted residuals 
    and if the residual is correlated with column A, we mark a YES in this case and a NO if not correlated
    2. You then fit the linear regression with column B as the feature and A as the outcome variable. Collect the fitted residuals 
    and if the residual is correlated with column B, we mark a YES in this case and a NO if not correlated

    If 1 is a YES and 2 is a NO, we say that B causes A; if 1 is a NO and 2 is a YES, we say that A causes B. \n
    Following above instruction to suggest causal pairs with direction among following variables after analyzing following data:\n{}. \n MUST Suggest 
ONLY the directed causal pairs without saying any other things:
    """.format(data.to_string())

    prediction = llm.predict(prompt1)

    prompt2 = """
    Here is the predicted pair:{} and here is the correct pair {}.
    If it is predicted correctly, return 1; if not, return 0;
    ONLY RETURN ME THE NUMBER 1 or 0.
    """.format(prediction, true)

    result =  openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "system", "content": "You are a helpful assistant to following the instruction"},
    {"role": "user", "content": prompt2
        }],
        temperature=0,
    )['choices'][0]['message']['content']

    return int(result)


    


