
import numpy as np
from torch.nn.functional import fold
from torch.serialization import load
from torch.types import Number
from torch.utils.data import Dataset
import logging
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import random
import torch
from tqdm import tqdm
LOGGER = logging.getLogger(__name__)

from .preprocesser import TextPreprocess


#candidate的来源是dictionary，用query查dict
class CandidateDataset(Dataset):
    """
    Candidate Dataset for:
        query_tokens, candidate_tokens, label
    """
    def __init__(self, queries, dicts, tokenizer, topk, d_ratio, s_score_matrix, s_candidate_idxs):
        """
        Retrieve top-k candidates based on sparse/dense embedding

        Parameters
        ----------
        queries : list
            A list of tuples (name, id)
        dicts : list
            A list of tuples (name, id)
        tokenizer : BertTokenizer
            A BERT tokenizer for dense embedding
        topk : int
            The number of candidates
        d_ratio : float
            The ratio of dense candidates from top-k
        s_score_matrix : np.array
        s_candidate_idxs : np.array
        """
        LOGGER.info("CandidateDataset! len(queries)={} len(dicts)={} topk={} d_ratio={}".format(
            len(queries),len(dicts), topk, d_ratio))
        self.query_names, self.query_ids = [row[0] for row in queries], [row[1] for row in queries]
        self.dict_names, self.dict_ids = [row[0] for row in dicts], [row[1] for row in dicts]
        self.topk = topk
        self.n_dense = int(topk * d_ratio)
        self.n_sparse = topk - self.n_dense
        self.tokenizer = tokenizer

        self.s_score_matrix = s_score_matrix
        self.s_candidate_idxs = s_candidate_idxs
        self.d_candidate_idxs = None

    def set_dense_candidate_idxs(self, d_candidate_idxs):
        self.d_candidate_idxs = d_candidate_idxs
    
    #获取对应元素的时候会自动利用更新好的dense_candidate
    def __getitem__(self, query_idx):
        assert (self.s_candidate_idxs is not None)
        assert (self.s_score_matrix is not None)
        assert (self.d_candidate_idxs is not None)

        query_name = self.query_names[query_idx]
        query_token = self.tokenizer.transform([query_name])

        # combine sparse and dense candidates as many as top-k
        s_candidate_idx = self.s_candidate_idxs[query_idx]
        d_candidate_idx = self.d_candidate_idxs[query_idx]
        
        # fill with sparse candidates first
        topk_candidate_idx = s_candidate_idx[:self.n_sparse]
        
        # fill remaining candidates with dense，逐项加入，不考虑overlap的情况
        for d_idx in d_candidate_idx:
            if len(topk_candidate_idx) >= self.topk:
                break
            if d_idx not in topk_candidate_idx:
                topk_candidate_idx = np.append(topk_candidate_idx,d_idx)
        
        # sanity check
        assert len(topk_candidate_idx) == self.topk
        assert len(topk_candidate_idx) == len(set(topk_candidate_idx))#保证没有重复
        
        candidate_names = [self.dict_names[candidate_idx] for candidate_idx in topk_candidate_idx]
        candidate_s_scores = self.s_score_matrix[query_idx][topk_candidate_idx]
        labels = self.get_labels(query_idx, topk_candidate_idx).astype(np.float32)
        query_token = np.array(query_token).squeeze()

        candidate_tokens = self.tokenizer.transform(candidate_names)
        candidate_tokens = np.array(candidate_tokens)
        
        return (query_token, candidate_tokens, candidate_s_scores), labels#返回了sparse score

    def __len__(self):
        return len(self.query_names)

    def check_label(self, query_id, candidate_id_set):
        label = 0
        query_ids = query_id.split("|")
        """
        All query ids should be included in dictionary id
        """
        for q_id in query_ids:
            if q_id in candidate_id_set:
                label = 1
                continue
            else:
                label = 0
                break
        return label

    def get_labels(self, query_idx, candidate_idxs):
        labels = np.array([])
        query_id = self.query_ids[query_idx]
        #print("query_id",query_id)
        candidate_ids = np.array(self.dict_ids)[candidate_idxs]
        #print("candidate_ids",candidate_ids)
        for candidate_id in candidate_ids:
            label = self.check_label(query_id, candidate_id)
            labels = np.append(labels, label)
        return labels#返回一个1/0来表示query和对应的candidate是否表达一个index

#given single file, construct corresponding graph of terms and its dictionary and query set
def load_data(filename='../../../data/datasets/cl.obo',migration_rate = 0.5, use_text_preprocesser = True):
    """
    some pre process rules:
    1.To oavoid overlapping, we just abandon the synonyms which are totally same as their names
    2. Considering that some names appear twice or more, We abandon correspoding synonyms
    3.Some synonyms have more than one corresponding term, we just take the first time counts
    """
    text_processer = TextPreprocess() 
    name_list = []#record of all terms, rememeber some elements are repeated
    sorted_name_set = None# record of all the terms' names. no repeated element, in the manner of lexicographic order
    mention2id = {}# map all mentions(names and synonyms of all terms) to ids, the name and synonyms with same term have the same id
    id2mention_group={}#map ids to their corresponding name and synonyms
    dict_set,query_set = [],[]# list of （mention,id）, in the order of ids
    edges=[] #list of tuples whose format is like(a,b), where a and b indicate the id of father_node and son_node respectively

    with open(file=filename,mode='r',encoding='utf-8') as f:
        check_new_term = False
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] and ends with a '\n'
                check_new_term = True
                continue
            if line[:1]=='\n':
                check_new_term = False
                continue
            if check_new_term == True:
                if line[:5]=='name:':
                    name_list.append(text_processer.run(line[6:-1])if use_text_preprocesser else line[6:-1])
        
        name_count = {}
        
        #record the count of names in raw file
        for i,name in enumerate(name_list):
            name_count[name] = name_list.count(name)
        
        #build a mapping function of name2id, considering that some names appear twice or more, we remove the duplication and sort them
        sorted_name_set = sorted(list(set(name_list)))
        dict_set = [(name,i) for i,name in enumerate(sorted_name_set)]

        for i,name in enumerate(sorted_name_set):
            mention2id[name] = i
        
        #temporary variables for every term
        #construct a scipy csr matrix of edges and collect synonym pairs
        check_new_term = False
        check_new_name = False#remember that not every term has a name and we just take the terms with name count. Good news: names' locations are relatively above
        synonym_group = []#record the (synonym,id) of  current term
        name = ""
        iter_name = iter(name_list)


        for i,line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] and ends with an '\n'
                check_new_term = True
                continue
            if line[:5]=='name:':

                check_new_name = True
                if check_new_term == True:
                    name = next(iter_name)
                continue
            if line[:1]=='\n':# signal the end of current term, deal with the synonym_group to construct the dictionary_set and query_set
                if check_new_term == True and check_new_name == True:
                    id = mention2id[name]
                    id2mention_group[id] = [name] + synonym_group# the first element will be name
                    
                    #split the symonym_group to dictionary and query
                    migration_num = int(len(synonym_group) * migration_rate)
                    dict_data,query_data = random_split(synonym_group, [migration_num, len(synonym_group) - migration_num], generator=torch.Generator().manual_seed(0)) 
                    dict_set+=[synonym_group[_] for _ in dict_data.indices]
                    query_set+=[synonym_group[_] for _ in query_data.indices]
                    

                check_new_term = False
                check_new_name = False
                synonym_group = []
                continue

            if check_new_term == True and check_new_name == True:
                #construct term graph
                if line[:5]=='is_a:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        father_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if father_node in sorted_name_set:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((mention2id[father_node],mention2id[name]))
                
                # collect synonyms and to dictionary set and query set
                if line[:8]=='synonym:' and name_count[name] == 1: #anandon the situations that name appears more than once
                    start_pos = line.index("\"") + 1
                    end_pos = line[start_pos:].index("\"") + start_pos
                    synonym = text_processer.run(line[start_pos:end_pos]) if use_text_preprocesser else line[start_pos:end_pos]
                    if synonym==name:continue#filter these mentions that are literally equal to the node's name,make sure there is no verlap
                    if synonym in mention2id.keys():continue# only take the first time synonyms appears counts
                    id = mention2id[name]
                    synonym_group.append((synonym,id))
                    mention2id[synonym] = id
        
        dict_set = sorted(dict_set,key = lambda x:x[1])
        query_set = sorted(query_set,key = lambda x:x[1])
        """
        print(len(mention2id.items()))
        mentions = [x for group in id2mention_group.values() for x in group ]
        print(len(mentions))
        print(len(name_list))
        print(id2mention_group[0])
        print(len(query_set))
        print(len(dict_set))
        """

        return sorted_name_set,mention2id,id2mention_group,dict_set,query_set,edges




#split training and test data for one file that corresponds to the queries
def data_split(queries,is_unseen=True,test_size = 0.33,folds = 1,seed = 0):
    """
    args:
    is_unseen:if is_unseen==true, then the ids in training pairs and testing pairs will not overlap 
    returns:
    three folds of train,test datasets
    """
    datasets_folds=[]
    random.seed(seed)
    np.random.seed(seed)
    #notice that we collect the (mention,concept) pairs in a order of all the concepts, so the same concepts will assemble together
    #as a result, we could remove all the (mention,concept) pairs with the same concept in an easy manner 
    mentions = [mention for (mention,id) in queries] 
    ids = [id for (mention,id) in queries]
    
    
    #random split
    if is_unseen == False:
        for fold in range(folds):
            mentions_train,mentions_test,ids_train,ids_test = train_test_split(
                mentions,ids,test_size=test_size)#have already set up seed 

            queries_train = [(mentions_train[i],ids_train[i]) for i in range(len(mentions_train))]
            queries_test = [(mentions_test[i,ids_test[i]]) for i in range(len(mentions_test))]
            datasets_folds.append((queries_train,queries_test))
    
    #random split, and the concepts in train set and test set will not overlap
    else:
        for fold in range(folds):
            mentions_train,mentions_test,ids_train,ids_test=mentions.copy(),[],ids.copy(),[]
            
            left_ids = sorted(list(set(ids)))
            while len(mentions_test) < len(mentions) * test_size:
                id = random.sample(left_ids,1)[0]
                
                start_index,end_index = ids.index(id), len(ids)-1 -  list(reversed(ids)).index(id)#the start index and the end index of the same concept

                for K in range(start_index,end_index+1):
                    mentions_test.append(mentions[K])
                    mentions_train.remove(mentions[K])
                    ids_test.append(id)
                    ids_train.remove(id)
                
                left_ids.remove(id)

            queries_train = [(mentions_train[i],ids_train[i]) for i in range(len(mentions_train))]
            queries_test = [(mentions_test[i],ids_test[i]) for i in range(len(mentions_test))]
            datasets_folds.append((queries_train,queries_test))

            #check overlap
            #for concept in concepts_test:
            #    if concept in concepts_train:
            #        print(concept)
                
    return datasets_folds


class MyDataset(Dataset):
    def __init__(self,filename='../../../data/datasets/cl.obo',migration_rate = 0.5,use_text_preprocesser= False):
        """
        args:
            filename:obo文件
            migration_rate:synonyms迁移到dict之中的概率
        """
        super(MyDataset).__init__()
        self.filename = filename
        self.migration_rate = migration_rate
        self.use_text_preprocesser = use_text_preprocesser
        self.sorted_name_set,self.mention2id,self.id2mention_group,self.dict_set,self.query_set,self.edges = load_data(filename,migration_rate,use_text_preprocesser)
        
    def load_dict_data(self):
        return np.array(self.dict_set)
    def load_query_data(self,mode='train',is_unseen=True,test_size = 0.33,folds = 1,seed = 0):
        data_folds = data_split(self.query_set,is_unseen=is_unseen,test_size=test_size,folds=folds,seed=seed)
        query_folds = [i for i,_ in data_folds] if mode=='train'else [i for _,i in data_folds]
        if folds==1:query_folds = query_folds[0]
        return np.array(query_folds)


        



#load_data(filename='../../../data/datasets/cl.obo',migration_rate = 0.5,use_text_preprocesser= False)