import pandas as pd
import os
import networkx as nx
import openai
import numpy as np
import spacy
import string
import re
from tqdm import tqdm
import string

error_word = {'neighbour': 'neighbor',
              'neighbouring': 'neighbor',
              'written': 'write',
              'spoken': 'speak'}

def dfs_triples_limit(df, start_node, max_triples = 40, reverse=False, visited=None, results=None):
    if visited is None:
        visited = set()
    if results is None:
        results = []
        
    # print(f"Starting or Continuing DFS from node: {start_node}, Visited: {visited}")

    # 如果达到最大三元组数量限制，则停止递归
    if len(results) >= max_triples:
        return

    visited.add(start_node)

    # 根据 reverse 参数选择搜索方向
    triples = df[df['tail' if reverse else 'head'] == start_node]
    
    # print(f"Found {len(triples)} triples from node: {start_node}")

    for _, row in triples.iterrows():
        # 检查是否已经达到最大三元组数量
        if len(results) < max_triples:
            node_to_visit = row['head' if reverse else 'tail']
            if node_to_visit not in visited:
                # print(f"Visiting node: {node_to_visit}, Current results: {len(results)}")
                results.append(row)
                dfs_triples_limit(df, node_to_visit, max_triples, reverse, visited, results)
                # print(f"Returning from node: {node_to_visit}, Total results: {len(results)}")
        else:
            # print("Reached max triples limit during recursion.")
            break
    
    return pd.DataFrame(results)
    
    # 如果是初始调用，返回结果 DataFrame
    # if visited == {start_node}:
    #     # print("Initial call returning, final results:", len(results))
    #     result_df = pd.DataFrame(results)
    #     return result_df

def bfs_node_limit(df, start_node, max_nodes, reverse=False):
    visited = {start_node}
    queue = [start_node]
    results = []

    while queue and len(visited) <= max_nodes:
        current_node = queue.pop(0)

        # 根据 reverse 参数选择搜索方向
        triples = df[df['tail' if reverse else 'head'] == current_node]

        for _, row in triples.iterrows():
            node_to_visit = row['head' if reverse else 'tail']
            if node_to_visit not in visited and len(visited) <= max_nodes:
                visited.add(node_to_visit)
                queue.append(node_to_visit)
                results.append(row)

    result_df = pd.DataFrame(results)
    return result_df


def bfs_depth_limit(df, start_node= None, reverse = False, max_depth =0):
    visited = {start_node}
    queue = [(start_node, 0)]
    results = []

    while queue:
        current_node, current_depth = queue.pop(0)

        if current_depth > max_depth:
            break

        # 根据 reverse 参数选择搜索方向
        triples = df[df['tail' if reverse else 'head'] == current_node]

        for _, row in triples.iterrows():
            node_to_visit = row['head' if reverse else 'tail']
            if node_to_visit not in visited:
                visited.add(node_to_visit)
                if current_depth + 1 <= max_depth:
                    queue.append((node_to_visit, current_depth + 1))
                results.append(row)

    result_df = pd.DataFrame(results)
    return result_df


def write_prompt(prompt, path, head, tail, relation, Positive = True):
    prompts_dir = os.path.join(path, 'prompts')
    
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)
    
    if Positive:
        if not os.path.exists(os.path.join(prompts_dir, 'positive')):
            os.makedirs(os.path.join(prompts_dir, 'positive'))
        prompts_dir = os.path.join(prompts_dir, 'positive')
    else:
        if not os.path.exists(os.path.join(prompts_dir, 'negative')):
            os.makedirs(os.path.join(prompts_dir, 'negative'))
        prompts_dir = os.path.join(prompts_dir, 'negative')
    
    file_name = f"prompt_{head}_{relation}_{tail}.txt"
    file_path = os.path.join(prompts_dir, file_name)
    
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(prompt)


def get_info(df,codex):
    prompt_information_provided = ""
    for _, row in df.iterrows():
        ehead = row['head']
        erel = row['relation']
        etail = row['tail']
        prompt_information_provided += f"head:{codex.entity_label(ehead)}, relation:{codex.relation_label(erel)}, tail:{codex.entity_label(etail)}.\n"
    return prompt_information_provided


def triple_TF(codex,head,relation,tail, retrieval = bfs_depth_limit, **kwargs):
    head_label = codex.entity_label(head)
    tail_label = codex.entity_label(tail)
    rel_label = codex.relation_label(relation)
    train = codex.split('train')
    try:
        if train[train["head"] == head].empty:
            raise ValueError(f"No tuples in train set with head {head_label}")
        head_part_df = retrieval(train, head, reverse = False, **kwargs)
        prompt_information_provided_head = get_info(head_part_df, codex)
    except ValueError as e:
        print(e)
        prompt_information_provided_head = ""
    
    try:
        if train[train["tail"] == tail].empty:
            raise ValueError(f"No tuples in train set with tail {tail_label}")
        tail_part_df = retrieval(train, tail, reverse=True, **kwargs)
        prompt_information_provided_tail = get_info(tail_part_df, codex)
    except ValueError as e:
        print(e)
        prompt_information_provided_tail = ""
    
    prompt = f"Based on everything you know and given the following information:\n{prompt_information_provided_head}\nAnd the following information:\n{prompt_information_provided_tail}\nIs the following statement true or false: head:{head_label}, relation:{rel_label}, tail:{tail_label}.\n"
    return prompt

def find_paths_df_multi(codex, train, test_head, test_relation,test_tail):
    # 使用MultiDiGraph来允许多重边
    G = nx.MultiDiGraph()

    # 添加边，每个关系作为不同的边
    for index, row in train.iterrows():
        G.add_edge(row['head'], row['tail'], key=row['relation'], relation=row['relation'])
    
    print('Graph created')

    # 寻找所有路径的内部函数
    def find_paths(G, start, end):
        return list(nx.all_simple_paths(G, start, end))

    # 获取所有路径
    print('Finding paths')
    paths = find_paths(G, test_head, test_tail)

    print('Paths found')
    # 将路径转换成DataFrame的内部函数
    def paths_to_df(paths):
        all_rows = []
        for path in paths:
            for i in range(len(path) - 1):
                head = path[i]
                tail = path[i + 1]
                # 对于每一对节点，查找它们之间的所有关系
                for key in G[head][tail]:
                    relation = G[head][tail][key]['relation']
                    all_rows.append({'head': head, 'relation': relation, 'tail': tail})
        return pd.DataFrame(all_rows)

    # 转换后的DataFrame
    res = paths_to_df(paths)
    if res.empty:
        prompt_information_provided = " "
    prompt_information_provided = get_info(res, codex)
    prompt = f"Given the following information:\n{prompt_information_provided}\nIs the following statement true or false: head:{codex.entity_label(test_head)}, relation:{codex.relation_label(test_relation)}, tail:{codex.entity_label(test_tail)}.\n"
    return prompt

def get_response(system_message, prompt, assistant_message = None):
    # API key
    client = openai.OpenAI(api_key='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')

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
    client = openai.OpenAI(api_key='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')

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

# def lemmatize_word(phrase, lemmatizer):
#     return lemmatizer(phrase)

def word_embedding(model, single_phrase = None, phrase_list = None):
    # model = KeyedVectors.load_word2vec_format('/Users/weichentao/Documents/pre_trained_models/GoogleNews-vectors-negative300.bin', binary=True)
    def phrase_vector(phrase, model):
        words = phrase.split()
        # word_vectors = [model[word] for word in words if word in model]
        word_vectors = []
        mls_cnt = 0
        for word in words:
            if word in model:
                word_vectors.append(model[word])
            elif word in error_word:
                word_vectors.append(model[error_word[word]])
            else:
                mls_cnt += 1
                # print(f"Word not found in model: {word}")
        if not word_vectors:
            return np.zeros(model.vector_size)
        else:
            return np.mean(word_vectors, axis=0) if mls_cnt == 0 else np.float32(sum(word_vectors) / (len(words) - mls_cnt))
    
    if single_phrase:
        return phrase_vector(single_phrase, model)
    else:
        phrase_vectors = [phrase_vector(phrase, model) for phrase in phrase_list]
        return phrase_vectors

def choose_most_similar(all_phrases_vec, target_phrase_vec, all_phrases):
    most_similar = None
    best_similarity = -1
    for i, phrase_vec in enumerate(all_phrases_vec):
        # print(f"Type of phrase_vec: {type(phrase_vec)}, Shape: {phrase_vec.shape}")  # 打印类型和形状
        # print(f"Type of target_phrase_vec: {type(target_phrase_vec)}, Shape: {target_phrase_vec.shape}")  # 打印类型和形状
        similarity = np.dot(phrase_vec, target_phrase_vec) / (np.linalg.norm(phrase_vec) * np.linalg.norm(target_phrase_vec))
        # print(f"Similarity: {similarity}")
        if type(similarity) != np.float32:
            print(all_phrases[i])
            print(f"Similarity: {similarity}")
        if similarity > best_similarity:
            best_similarity = similarity
            most_similar = all_phrases[i]
    return most_similar

def preprocess_phrase(phrase, nlp):
    p_phrase = phrase.lower()  # Convert all letters to lowercase
    p_phrase = p_phrase.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    p_phrase = re.sub(r'\s+', ' ', p_phrase)
    doc = nlp(p_phrase)
    p_phrase = ' '.join([token.lemma_ for token in doc])
    return p_phrase

def save_response(df, system_message, assistant_message = None, path = None):
    res = []
    # 使用tqdm(df.iterrows(), total=df.shape[0])来创建一个进度条
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        if index % 2 == 0:  # 使用index % 2 == 0来判断是否为偶数行
            response = get_response(system_message=system_message, prompt=row['prompt'])
        else:
            response = get_response_2(system_message=system_message, prompt=row['prompt'])
        res.append(response)
    
    df['response'] = res
    
    if path:  # 如果提供了路径，则保存到该路径
        df.to_csv(path, index=False)
    
    return df

def preprocess_df(df, codex):
    df = df.copy()
    df[['head_id','relation_id','tail_id']] = df[['head','relation','tail']]
    df[['head', 'relation', 'tail']] = df[['head', 'relation', 'tail']].apply(lambda x: x.apply(codex.entity_label if x.name == 'head' or x.name == 'tail' else codex.relation_label))
    df['prompt'] = df.apply(lambda x: f"What is the relationship between '{x['head']}' and '{x['tail']}'?", axis=1)
    return df


def extract_pred(response):
    response = response.lower()
    pred = []
    punctuation = string.punctuation.replace(',', '')
    trans_table = str.maketrans('', '', punctuation)
    if "relationship is ambiguous" in response:
        response = response.replace("relationship is ambiguous", "[relationship is ambiguous]")
    
    matches = re.findall(r'\[(.*?)\]', response)
    if not matches:
        pred.append("relationship is ambiguous")
    else:
        cleaned_matches = [s.translate(trans_table) for s in matches]
        pred.extend(cleaned_matches)
    return pred


def ask_twice(df,system_message = None):
    cnt = 0
    res = []
    if 'pred_relation' not in df.columns:
        print("No 'pred_relation' column found in DataFrame. Please run the model first.")
        return
    if not system_message:
        with open('/Users/weichentao/Documents/USC/research with mk/dataset/Codex/codex/prompts/system_messages/system_message_triple_verification.txt', 'r') as f:
            system_message = f.read()
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        cnt += 1
        result = []
        for pred in row['pred_relation']:
            # print(pred)
            if pred == 'relationship is ambiguous':
                result.append('relationship is ambiguous')
            else:
                prompt = f"Is the relationship True or False? {row['head']} [{pred}] {row['tail']}."
                if cnt % 2 == 0:    
                    response = get_response(system_message=system_message, prompt= prompt)
                else:
                    response = get_response_2(system_message=system_message, prompt= prompt)
                result.append(response)
        res.append(result)
    df['verification'] = res
    return df

def find_similar(entity_id, train, codex, k=5):
    df = train[train['head'] == entity_id]
    if len(df) <= k:
        return preprocess_df(df, codex)
    else:
        df = df.sample(k)
        return preprocess_df(df, codex)
    
    
# if __name__ == "__main__":
#     nlp = spacy.load("en_core_web_sm")
#     word = 'running on a playground'
#     for token in nlp(word):
#         print(token.text, token.lemma_)




    



    



        
        
        
            

