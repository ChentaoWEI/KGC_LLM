import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import sys
import ast
root = os.getcwd()
sys.path.append(root)

from dataset import WN18RR
from utils import reask, process_verification
import re
import random
from tqdm import tqdm
from const import WN18RR_relationship_list

if __name__ == '__main__':
    
    exp_list = ['zero_shot_one_relation', 'zero_shot_multi_relation', 'few_shot_one_relation', 'few_shot_multi_relation', 'zero_cot_one_relation', 'zero_cot_multi_relation']
    
    for exp_name in exp_list:
        print(f"Current experiment: {exp_name}")
        result_path = os.path.join('results/WN18RR', exp_name, 'result.csv')
        system_message_path = os.path.join('results/WN18RR', exp_name, 'system_message.txt')
        if exp_name in ['few_shot_one_relation', 'few_shot_multi_relation']:
            assistant_message_path = os.path.join('results/WN18RR', exp_name, 'assistant_message.txt')
        else:
            assistant_message_path = None
        save_path = os.path.join('results/WN18RR', exp_name, 'result.csv')
        df = pd.read_csv(result_path)
        
        df.verification = df.verification.apply(lambda x: ast.literal_eval(x))
        df.pred_relation = df.pred_relation.apply(lambda x: ast.literal_eval(x))
        
        if 'pred_relation_2' in df.columns:
            print('Already asked twice')
        else:
            df = process_verification(df)
            
            df = reask(df, WN18RR_relationship_list,
                    system_message_path = system_message_path, 
                    assistant_message_path = assistant_message_path, 
                    save_path = save_path)
            
            print("Done!")
            print(df.head())
        
        