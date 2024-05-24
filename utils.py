import pandas as pd
import openai
import string
import re
from tqdm import tqdm
import math
from transformers import GPT2Tokenizer

api = 'your key'

def count_tokens(system_message, assistant_message):
    # 创建一个GPT-2的分词器
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 将system_message和assistant_message拼接为一个完整的prompt
    prompt = system_message + assistant_message
    
    # 使用tokenizer对prompt进行分词，计算token的数量
    tokens = tokenizer.encode(prompt)
    num_tokens = len(tokens)

    return num_tokens

def preprocess_WN(df):
    df['head'] = df['head'].apply(lambda x: x.split('.')[0])
    df['tail'] = df['tail'].apply(lambda x: x.split('.')[0])
    df['relation'] = df['relation'].apply(lambda x: x.replace('_', ' ').strip())
    return df

def get_response(system_message, prompt, assistant_message = None):
    # API key
    client = openai.OpenAI(api_key='your key')

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": assistant_message},
            {"role": "user", "content": prompt}
        ] if assistant_message else [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def get_response_2(system_message, prompt, assistant_message = None):
    # API key
    client = openai.OpenAI(api_key='your key')

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": assistant_message},
            {"role": "user", "content": prompt}
        ] if assistant_message else [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def get_response_gpt4(system_message, prompt, assistant_message = None):
    # API key
    client = openai.OpenAI(api_key=api)


    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": assistant_message},
            {"role": "user", "content": prompt}
        ] if assistant_message else [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def get_user_response(system_message, prompt, assistant_message, gpt4, response_alternation_count):
    """Get response based on the GPT model used and the alternation count."""
    if gpt4:
        return get_response_gpt4(system_message=system_message, prompt=prompt, assistant_message=assistant_message)
    else:
        # Alternates between two response functions
        if response_alternation_count % 2 == 0:
            return get_response(system_message=system_message, prompt=prompt, assistant_message=assistant_message)
        else:
            return get_response_2(system_message=system_message, prompt=prompt, assistant_message=assistant_message)



def extract_pred(response):
    response = response.lower()
    pred = []
    punctuation = string.punctuation.replace(',', '')
    trans_table = str.maketrans('', '', punctuation)
    if "relationship is ambiguous" in response:
        response = response.replace("relationship is ambiguous", "[relationship is ambiguous]")
    
    matches = re.findall(r'\[(.*?)\]', response)
    if not matches:
        pred.append("relationship not found")
    else:
        cleaned_matches = [s.translate(trans_table) for s in matches]
        pred.extend(cleaned_matches)
    return pred

def preprocess_df(df, codex = None):
    # Generate prompt for each triple.
    df = df.copy()
    if not codex:
        df['prompt'] = df.apply(lambda x: f"What is the relationship between '{x['head']}' and '{x['tail']}'?", axis=1)
    else:  
        df[['head_id','relation_id','tail_id']] = df[['head','relation','tail']]
        df[['head', 'relation', 'tail']] = df[['head', 'relation', 'tail']].apply(lambda x: x.apply(codex.entity_label if x.name == 'head' or x.name == 'tail' else codex.relation_label))
    df['prompt'] = df.apply(lambda x: f"What is the relationship between '{x['head']}' and '{x['tail']}'?", axis=1)
    return df

def ask_first(df, relation_list,system_message_path, assistant_message_path = None, assistant_message = None, save_path = None, gpt4 = False):
    with open(system_message_path, 'r') as file:
        system_message = file.read()
    if not assistant_message_path and not assistant_message:
        assistant_message = None
    elif assistant_message_path and not assistant_message:
        with open(assistant_message_path, 'r') as file:
            assistant_message = file.read()
    elif assistant_message and not assistant_message_path:
        assistant_message = assistant_message
            
    system_message = system_message + f'You should choose your answer from the relation set:{list(relation_list)}'
    
    print("Asking...")
    print(system_message)
    print(assistant_message)

    num = 1
    res = []
    # shuffled_df = get_df(codex, pos = False, seed = 42, pos_size = 500, neg_size = 500)
    

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        # print(row['prompt'])
        response = get_user_response(system_message=system_message, prompt=row['prompt'], assistant_message=assistant_message, gpt4=gpt4, response_alternation_count=num)
        num += 1
        res.append(response)
    df['response'] = res
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

def ask_twice(df, system_message_path, assistant_message_path,save_path, gpt4 = False): 
    with open(system_message_path, 'r') as f:
        system_message = f.read()
    if assistant_message_path:
        with open(assistant_message_path, 'r') as f:
            assistant_message = f.read()
    else:
        assistant_message = None
    cnt = 0
    res = []
    
    print("Verifying...")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        cnt += 1
        result = []
        if not row['pred_relation']:
            continue
        for pred in row['pred_relation']:
            # print(pred)
            if pred == 'relationship is ambiguous':
                result.append('relationship is ambiguous')
            else:
                prompt = f"Is the relationship True or False? {row['head']} [{[pred]}] {row['tail']}."
                response = get_user_response(system_message=system_message, prompt=prompt, assistant_message=assistant_message, gpt4=gpt4, response_alternation_count=cnt)
                cnt += 1
                result.append(response)
        res.append(result)
    df['verification'] = res
    df.to_csv(save_path, index=False)
    return df

def formulate_triple(row):
    return f"'{row['head']}' [{row['relation']}] '{row['tail']}'"

def process_verification(df):
    if 'pred_verification' in df.columns:
        print("Already processed")
        return df
    
    pred_verifications = []
    for ind, row in df.iterrows():
        result = []
        for i, ver in enumerate(row['verification']):
            match = re.search(r'^\w+', ver)
            if match:
                first_word = match.group()
                if str.lower(first_word) not in ['true','false']:
                    print(f"faliure in row {ind}, answer {i}")
                    result.append('not found')
                else:
                    result.append(str.lower(first_word))
            else:
                print(f"faliure in row {ind}, answer {i}")
                result.append('not found')
                
        pred_verifications.append(result)
    df['pred_verification'] = pred_verifications
    
    return df


def hit_at_n(df, pred_col, true_col, n):
    if pred_col in df.columns and true_col in df.columns:
        cnt = 0
        for ind, row in df.iterrows():
            if row[true_col] in row[pred_col][:n]:
                cnt += 1
        return cnt/len(df)
    else:
        print("No pred_relation or true_relation column in the dataframe")
        return None


def reask(df, relation_list, system_message_path, assistant_message_path = None, assistant_message = None,save_path = None, gpt4 = False):
    with open(system_message_path, 'r') as f:
        system_message = f.read()
    if not assistant_message_path and not assistant_message:
        assistant_message = None
    elif assistant_message_path and not assistant_message:
        with open(assistant_message_path, 'r') as file:
            assistant_message = file.read()
    elif assistant_message and not assistant_message_path:
        assistant_message = assistant_message
    
    print("Reasking...")   
    # print(system_message)
    print(assistant_message)
    
    num = 1
    res = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        if not row['pred_relation']:
            continue
        
        excluded_relations = []
        rels = []
        for ind, pred in enumerate(row['pred_relation']):
            if row['pred_verification'][ind] == 'false' and pred in relation_list:
                excluded_relations.append(pred)
            elif row['pred_verification'][ind] == 'true' and pred in relation_list:
                rels.append(pred) 
        
        if not excluded_relations:
            lst = relation_list
        else:
            lst = [x for x in relation_list if x not in excluded_relations]
        # print(f"excluded_relations: {excluded_relations}")
        # print(f'lst: {lst}')
        sys = system_message + f'You should choose your answer from the relation set:{list(lst)}'
        print(sys)
        response = get_user_response(system_message=sys, prompt=row['prompt'], assistant_message=assistant_message, gpt4=gpt4, response_alternation_count=num)
        num += 1
        rels.extend(extract_pred(response))
        res.append(rels)
        
    df['pred_relation_2'] = res
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

def process_codex_verification(df):
    
    df = df.copy()
    if 'pred_verification' not in df.columns:
        print("No pred_verification column in the dataframe")
    
    def process_verification(row):
        result = []
        for i, ver in enumerate(row['pred_verification']):
            if ver == 'relationship is ambiguous':
                result.append('relationship is ambiguous')
            else:
                if ver == 'true':
                    result.append(row['pred_relation'][i])
        if not result:
            result.append('relationship is ambiguous')
        row['pred_relation_2'] = result
        return row['pred_relation_2']

    df['pred_relation_2'] = None
    df['pred_relation_2'] = df.apply(process_verification, axis=1)
    
    return df

