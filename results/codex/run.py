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

from utils import ask_first, ask_twice ,preprocess_df, process_verification, extract_pred

from Codex.codex import Codex


def mid_process(df):
    # Extract the predicted relation from the response
    df['pred_relation'] = df['response'].apply(lambda x: extract_pred(x))
    # df.pred_relation = df.pred_relation.apply(lambda x: ast.literal_eval(x))
    df['true_relation'] = df['relation']
    
    return df

def get_df(codex, pos = False, seed = 42, pos_size = 500, neg_size = 500):
    train = codex.split('train')
    valid_pos = codex.split('valid')
    test_pos = codex.split('test')
    test_neg = codex.negative_split('test')
    valid_neg = codex.negative_split('valid')
    
    np.random.seed(seed)
    if pos:
        test_triple_pos = test_pos.sample(n=pos_size).squeeze()
        df = test_triple_pos
        pass
    else:
        test_triple_pos = test_pos.sample(n=pos_size).squeeze()
        test_triple_neg = test_neg.sample(n=neg_size).squeeze()
        test_triple_pos = test_triple_pos.assign(Label=1)
        test_triple_neg = test_triple_neg.assign(Label=0)
        df = pd.concat([test_triple_pos, test_triple_neg])
    
    shuffled_df = df.sample(frac=1, random_state=seed)
    shuffled_df = preprocess_df(shuffled_df, codex)
    
    return shuffled_df


if __name__ == '__main__':
    
    sizes = Codex.SIZES
    codes = Codex.CODES
    codex = Codex(size='s', code='en')
    
    all_relations = codex.relations()
    gpt4 = False
    
    zero_shot_list = ['zero_shot_one_relation', 'zero_shot_multi_relation', 'zero_cot_one_relation']
    few_shot_list = [] #  ['few_shot_multi_relation']
    root_path = 'results/codex'
    
    
    for exp_name in zero_shot_list:
        system_message_path = os.path.join(root_path, exp_name, 'system_message.txt')
        assistant_message_path = None
        save_path = os.path.join(root_path, exp_name, 'result.csv')
        verification_path = os.path.join(root_path, 'ask_twice_system_message.txt')
        verification_assistant = os.path.join(root_path, 'ask_twice_assistant_message.txt')
        
        df = get_df(codex, pos = False, seed = 42, pos_size = 500, neg_size = 500)
        
        df = ask_first(df, all_relations, system_message_path = system_message_path, assistant_message_path = assistant_message_path, save_path = save_path, gpt4 = gpt4)
        df = mid_process(df)

        df = ask_twice(df, system_message_path = verification_path, assistant_message_path = verification_assistant, save_path = save_path, gpt4 = gpt4)
        df = process_verification(df)
        df.to_csv(save_path, index=False)
        # df = reask(df, all_relations, system_message_path = system_message_path, assistant_message_path = assistant_message_path, save_path = save_path, gpt4 = gpt4)
        print("Done!")
    
    for exp_name in few_shot_list:
        system_message_path = os.path.join(root_path, exp_name, 'system_message.txt')
        
        assistant_message_path = os.path.join(root_path, exp_name, 'assistant_message.txt')
        
        save_path = os.path.join(root_path, exp_name, 'result.csv')
        verification_path = os.path.join(root_path, 'ask_twice_system_message.txt')
        verification_assistant = os.path.join(root_path, 'ask_twice_assistant_message.txt')

        
        df = get_df(codex, pos = False, seed = 42, pos_size = 500, neg_size = 500)
        
        df = ask_first(df, all_relations, system_message_path = system_message_path, assistant_message= None, assistant_message_path= assistant_message_path, save_path = save_path, gpt4 = gpt4)
        df = mid_process(df)

        df = ask_twice(df, system_message_path = verification_path, assistant_message_path = verification_assistant, save_path = save_path, gpt4 = gpt4)
        df = process_verification(df)
        df.to_csv(save_path, index=False)
        # df = reask(df, all_relations, system_message_path = system_message_path, assistant_message=assistant_message, save_path = save_path, gpt4 = gpt4)
        print("Done!")
    
    
    