import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import ast

import warnings
warnings.filterwarnings("ignore")
root = os.getcwd()
sys.path.append(root)

from utils import ask_first, ask_twice, preprocess_df, process_verification, reask, extract_pred

from dataset import Yago3_10


def mid_process(df):
    # Extract the predicted relation from the response
    df['pred_relation'] = df['response'].apply(lambda x: extract_pred(x))
    # df.pred_relation = df.pred_relation.apply(lambda x: ast.literal_eval(x))
    df['true_relation'] = df['relation']
    
    return df

if __name__ == '__main__':
    yago = Yago3_10()
    yago.load()
    
    all_relations = yago.get_all_relations()
    gpt4 = True
    
    zero_shot_list = ['zero_shot_one_relation', 'zero_shot_multi_relation', 'zero_cot_one_relation']
    few_shot_list = [] #  ['few_shot_multi_relation']
    root_path = 'results/Yago3-10'
    
    for exp_name in zero_shot_list:
        system_message_path = os.path.join(root_path, exp_name, 'system_message.txt')
        # system_message_path = os.path.join('results/Yago3-10', exp_name, 'system_message.txt')
        assistant_message_path = None
        save_path = os.path.join(root_path, exp_name, 'result_gpt4.csv')
        # save_path = os.path.join('results/Yago3-10', exp_name, 'result_gpt4.csv')
        verification_path = os.path.join(root_path, 'ask_twice_system_message.txt')
        # verification_path = os.path.join('results/Yago3-10', 'ask_twice_system_message.txt')
        verification_assistant = os.path.join(root_path, 'ask_twice_assistant_message.txt')
        # verification_assistant = os.path.join('results/Yago3-10', 'ask_twice_assistant_message.txt')
        
        df = yago.test_.sample(n=100, random_state=42)
        df = preprocess_df(df)
        
        df = ask_first(df, all_relations, system_message_path = system_message_path, assistant_message_path = assistant_message_path, save_path = save_path, gpt4 = gpt4)
        df = mid_process(df)

        df = ask_twice(df, system_message_path = verification_path, assistant_message_path = verification_assistant, save_path = save_path, gpt4 = gpt4)
        df = process_verification(df)
        # df.verification = df.verification.apply(lambda x: ast().literal_eval(x))
        df = reask(df, all_relations, system_message_path = system_message_path, assistant_message_path = assistant_message_path, save_path = save_path, gpt4 = gpt4)
        print("Done!")
    
    for exp_name in few_shot_list:
        system_message_path = os.path.join(root_path, exp_name, 'system_message.txt')
        # system_message_path = os.path.join('results/Yago3-10', exp_name, 'system_message.txt')
        # assistant_message_path = os.path.join('results/Yago3-10', exp_name, 'assistant_message.txt')
        
        examples = yago.sample_high_frequence_triples(20, seed = 42)
        assistant_message = "Examples:\n"
        for idx, row in examples.iterrows():
            assistant_message += f"{idx+1}. {row['prompt']}\n"
            assistant_message += f"Reply: {row['answer']}\n\n"
        
        
        save_path = os.path.join(root_path, exp_name, 'result_gpt4.csv')
        # save_path = os.path.join('results/Yago3-10', exp_name, 'result_gpt4.csv')
        verification_path = os.path.join(root_path, 'ask_twice_system_message.txt')
        verification_assistant = os.path.join(root_path, 'ask_twice_assistant_message.txt')
        # verification_path = os.path.join('results/Yago3-10', 'ask_twice_system_message.txt')
        # verification_assistant = os.path.join('results/Yago3-10', 'ask_twice_assistant_message.txt')
        
        df = yago.test_.sample(n=100, random_state=42)
        df = preprocess_df(df)
        
        df = ask_first(df, all_relations, system_message_path = system_message_path, assistant_message= assistant_message, save_path = save_path, gpt4 = gpt4)
        df = mid_process(df)

        df = ask_twice(df, system_message_path = verification_path, assistant_message_path = verification_assistant, save_path = save_path, gpt4 = gpt4)
        df = process_verification(df)
        df = reask(df, all_relations, system_message_path = system_message_path, assistant_message=assistant_message, save_path = save_path, gpt4 = gpt4)
        print("Done!")
    
    
        