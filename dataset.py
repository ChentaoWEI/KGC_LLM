import os
import pandas as pd
import math
from utils import formulate_triple, preprocess_df
    
class WN18RR():
    COLUMNS = ['head', 'relation', 'tail']
    
    def __init__(self) -> None:
        self.relation_set = set()
        self.id2relation = {}
        self.relation_num = 18
        with open('WN18RR/relation2id.txt', 'r') as f:
            cnt = 0
            for line in f.readlines():
                if cnt == 0:
                    cnt += 1
                else:
                    rel = line.strip().split()[0][1:].replace('_', ' ')
                    self.relation_set.add(rel)
                    id = int(line.strip().split()[1])
                    self.id2relation[id] = rel
        
        self.train_, self.valid_, self.test_ = [pd.DataFrame() for _ in range(3)]
    
    def load(self):
        self.train_ = pd.read_csv('WN18RR/text/train.txt', sep='\t', names=self.COLUMNS)
        self.valid_ = pd.read_csv('WN18RR/text/valid.txt', sep='\t', names=self.COLUMNS)
        self.test_ = pd.read_csv('WN18RR/text/test.txt', sep='\t', names=self.COLUMNS)
        self.train_ = self.preprocess_WN(self.train_)
        self.valid_ = self.preprocess_WN(self.valid_)
        self.test_ = self.preprocess_WN(self.test_)
        
        print("Successfully loaded WN18RR dataset")
        
    def preprocess_WN(self, df):
        df['head'] = df['head'].apply(lambda x: x.split('.')[0])
        df['tail'] = df['tail'].apply(lambda x: x.split('.')[0])
        df['relation'] = df['relation'].apply(lambda x: x.replace('_', ' ').strip())
        return df

    def sample_high_frequence_triples(self, num, seed = 42):
        all_relations = self.relation_set
        train_df = self.train_
        
        # 计算每个关系的频率
        relation_counts = train_df['relation'].value_counts(normalize=True)
        
        # 计算每个关系应该抽样的数量，并向上取整
        samples_per_relation = (relation_counts * num).apply(math.ceil)
        
        # 抽样
        sampled_triples = pd.DataFrame()
        for relation, sample_size in samples_per_relation.items():
            relation_df = train_df[train_df['relation'] == relation]
            sampled = relation_df.sample(min(len(relation_df), sample_size), random_state=seed)
            sampled_triples = pd.concat([sampled_triples, sampled])
        
        # 打乱最终的DataFrame
        sampled_triples = sampled_triples.sample(frac=1, random_state=seed).reset_index(drop=True)
        sampled_triples = preprocess_df(sampled_triples)
        
        # 处理输出格式
        def process_answer(row):
            return f"'{row['head']}' [{row['relation']}] '{row['tail']}'"
        
        sampled_triples['answer'] = sampled_triples.apply(process_answer, axis=1)
        
        return sampled_triples
        

class Yago3_10():
    COLUMNS = ['head', 'relation', 'tail']
    def __init__(self) -> None:
        self.relation_set = set()
        self.relation_num = 0
        self.train_size = 0
        self.valid_size = 0
        self.test_size = 0
        self.loaded = False

        self.train_, self.valid_, self.test_ = [pd.DataFrame() for _ in range(3)]
     
    def load(self):
        self.train_ = pd.read_csv('other/Yago3-10/train.txt', sep='\t', names=self.COLUMNS)
        self.valid_ = pd.read_csv('other/Yago3-10/valid.txt', sep='\t', names=self.COLUMNS)
        self.test_ = pd.read_csv('other/Yago3-10/test.txt', sep='\t', names=self.COLUMNS)
        
        self.train_['head'] = self.train_['head'].str.replace('[-_]', ' ', regex=True)
        self.train_['tail'] = self.train_['tail'].str.replace('[-_]', ' ', regex=True)
        self.train_['relation'] = self.train_['relation'].str.replace('(?<!^)(?=[A-Z])', ' ', regex=True)
        self.train_['relation'] = self.train_['relation'].str.lower()
        # self.train_['relation'] = self.train_['relation'].str.replace('[-_]', ' ', regex=True)
        
        self.valid_['head'] = self.valid_['head'].str.replace('[-_]', ' ', regex=True)
        self.valid_['tail'] = self.valid_['tail'].str.replace('[-_]', ' ', regex=True)
        self.valid_['relation'] = self.valid_['relation'].str.replace('(?<!^)(?=[A-Z])', ' ', regex=True)
        self.valid_['relation'] = self.valid_['relation'].str.lower()
        # self.valid_['relation'] = self.valid_['relation'].str.replace('[-_]', ' ', regex=True)
        
        self.test_['head'] = self.test_['head'].str.replace('[-_]', ' ', regex=True)
        self.test_['tail'] = self.test_['tail'].str.replace('[-_]', ' ', regex=True)
        self.test_['relation'] = self.test_['relation'].str.replace('(?<!^)(?=[A-Z])', ' ', regex=True)
        self.test_['relation'] = self.test_['relation'].str.lower()
        
        print("Successfully loaded Yago3-10 dataset")
        self.loaded = True
        # self.test_['relation'] = self.test_['relation'].str.replace('[-_]', ' ', regex=True)
    
    def get_train_size(self):
        line_num = 0
        with open('other/Yago3-10/train.txt', 'r') as f:
            for line in f.readlines():
                line_num += 1
        self.train_size = line_num
        return line_num
    
    def get_valid_size(self):
        line_num = 0
        with open('other/Yago3-10/valid.txt', 'r') as f:
            for line in f.readlines():
                line_num += 1
        self.valid_size = line_num
        return line_num
    
    def get_test_size(self):
        line_num = 0
        with open('other/Yago3-10/test.txt', 'r') as f:
            for line in f.readlines():
                line_num += 1
        self.test_size = line_num
        return line_num
    
    def get_all_relations(self):
        if self.relation_set:
            return self.relation_set
        
        if not self.loaded:
            self.load()
        train_relation = set(self.train_['relation'].unique())
        valid_relation = set(self.valid_['relation'].unique())
        test_relation = set(self.test_['relation'].unique())
        self.relation_set = train_relation.union(valid_relation).union(test_relation)
        self.relation_num = len(self.relation_set)
        
        return self.relation_set

    def sample_high_frequence_triples(self, num, seed = 42):
        all_relations = self.get_all_relations() if not self.relation_set else self.relation_set
        train_df = self.train_
        
        # 计算每个关系的频率
        relation_counts = train_df['relation'].value_counts(normalize=True)
        
        # 计算每个关系应该抽样的数量，并向上取整
        samples_per_relation = (relation_counts * num).apply(math.ceil)
        
        # 抽样
        sampled_triples = pd.DataFrame()
        for relation, sample_size in samples_per_relation.items():
            relation_df = train_df[train_df['relation'] == relation]
            sampled = relation_df.sample(min(len(relation_df), sample_size), random_state=seed)
            sampled_triples = pd.concat([sampled_triples, sampled])
        
        # 打乱最终的DataFrame
        sampled_triples = sampled_triples.sample(frac=1, random_state=seed).reset_index(drop=True)
        sampled_triples = preprocess_df(sampled_triples)
        
        # 处理输出格式
        def process_answer(row):
            return f"'{row['head']}' [{row['relation']}] '{row['tail']}'"
        
        sampled_triples['answer'] = sampled_triples.apply(process_answer, axis=1)
        
        return sampled_triples
          

class WN18():
    COLUMNS_ORG = ['head', 'tail', 'relation']
    COLUMNS_TXT = ['head', 'relation', 'tail']
    def __init__(self) -> None:
        
        self.relation_set = set()
        self.id2relation = {}
        self.relation_num = 18
        with open('WN18/original/relation2id.txt', 'r') as f:
            cnt = 0
            for line in f.readlines():
                if cnt == 0:
                    cnt += 1
                else:
                    rel = line.strip().split()[0][1:].replace('_', ' ')
                    self.relation_set.add(rel)
                    id = int(line.strip().split()[1])
                    self.id2relation[id] = rel
        self.train_, self.valid_, self.test_ = [pd.DataFrame() for _ in range(3)]
    
    def load_origin(self):
        self.train_ = pd.read_csv('WN18/original/train.txt', sep=' ', names=self.COLUMNS_ORG, header = 1)
        self.valid_ = pd.read_csv('WN18/original/valid.txt', sep=' ', names=self.COLUMNS_ORG, header = 1)
        self.test_ = pd.read_csv('WN18/original/test.txt', sep=' ', names=self.COLUMNS_ORG, header = 1)
    
    def load(self):
        self.train_ = pd.read_csv('WN18/text/train.txt', sep='\t', names=self.COLUMNS_TXT)
        self.valid_ = pd.read_csv('WN18/text/valid.txt', sep='\t', names=self.COLUMNS_TXT)
        self.test_ = pd.read_csv('WN18/text/test.txt', sep='\t', names=self.COLUMNS_TXT)
    
    def get_train_num(self):
        with open('WN18/original/train.txt', 'r') as f:
            return int(f.readline())
    
    def get_valid_num(self):
        with open('WN18/original/valid.txt', 'r') as f:
            return int(f.readline())
    
    def get_test_num(self):
        with open('WN18/original/test.txt', 'r') as f:
            return int(f.readline())  
        

        
    
    
    
    