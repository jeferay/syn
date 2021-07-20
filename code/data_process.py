
import json
import wget
import os
import scipy
import  tqdm
import random
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

#catch all suitable datasets on the website
def get_all_data(filename='../data/ontologies.jsonld'):
    specific_problem_ids=['rs','fix','eo','envo']# for some unkonwn reasons, rs.obo, fix.obo and eo.obo can not be downloaded;and envo has a strange problem
    urls = []
    ids = []
    with open(filename,mode='r',encoding='utf-8') as f:
        content = json.load(f)['ontologies']
        for i,entry in enumerate(content):
            id = entry['id']
            #every entry has an id, and we only need to consider the urls which are normalized as {id}.obo
            if 'products' in entry.keys():
                products = entry['products']
                
                for product in products:
                    if product['id']==id + '.obo' and id not in specific_problem_ids:
                        urls.append(product['ontology_purl'])
                        ids.append(id)
    
    #download relative files to data_dir, finnally we get 95 files
    #print(ids)
    data_dir = '../data/datasets'
    for i,(id,url) in  enumerate(zip(ids,urls)):
        #print(id)
        filename = id+'.obo'
        file = wget.download(url=url,out= os.path.join(data_dir,filename))
    
#generate relative concepts ,construct a scipy csr matrix of edges to present the graph and collect synonym pairs for one file
def construct_graph(filename='../data/datasets/cl.obo'):
    """
    returns:
    concept_list:all the concepts
    concept2id:a dictionary that maps concept to its id
    edges: list of tuples whose format is like(a,b), where a and b indicate the id of father_node and son_node respectively
    mention_list: list of all the mentions that could be a synonym of one concept
    synonym_paris: list of tuples whose format is like (a,b), where a and b indicate the name of a mention and the name of its relative entity, respectively 
    """
    concept_list = []
    concept2id = {}
    mention_set = set()
    edges,synonym_pairs=[],[]
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
                    concept_list.append(line[6:-1])
        
        #build a mapping function
        for i,name in enumerate(concept_list):
            concept2id[name] = i
        
        #construct a scipy csr matrix of edges and collect synonym pairs
        check_new_term = False
        check_new_name = False#remember that not every term has a name and we just take the terms with name count. Good news: names' locations are relatively above
        node = ""
        iter_concept = iter(concept_list)
        for i,line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] and ends with an '\n'
                check_new_term = True
                continue
            if line[:5]=='name:':
                check_new_name = True
                if check_new_term == True:
                    node = next(iter_concept)
                continue
            if line[:1]=='\n':
                check_new_term = False
                check_new_name = False
                continue
            if check_new_term == True and check_new_name == True:
                if line[:5]=='is_a:':
                    entry = line.split(" ")
                    if '!' in entry:# some father_nodes are not divided by '!' and we abandon them
                        father_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                        if father_node in concept_list:#some father_nodes are not in concepts_list, and we abandon them.
                            edges.append((concept2id[father_node],concept2id[node]))
                    #print("--node--",node,'---father_node---',father_node)
                if line[:8]=='synonym:':
                    start_pos = line.index("\"") + 1
                    end_pos = line[start_pos:].index("\"") + start_pos
                    mention = line[start_pos:end_pos]
                    if mention==node:continue#filter these mentions that are literally equal to the node's name
                    mention_set.add(mention)
                    synonym_pairs.append((mention,node))
                    #print("--node--",node,'---synonym---',mention)
        
        return concept_list,concept2id,edges,list(mention_set),synonym_pairs

#generate negative samples if needed
def construct_positive_and_negative_pairs(concept_list,synonym_pairs,neg_posi_rate):
    """
    returns: positive pairs and negative pairs.And the number of negative samples is neg_posi_rate more than synonym pairs(positive samples)
    """
    negative_pairs = []
    for i,(mention,_) in enumerate(synonym_pairs):
        for _ in range(neg_posi_rate):
            concept = random.sample(concept_list,1)[0]
            while (mention,concept) in synonym_pairs or (mention,concept) in negative_pairs:#avoid overlapping
                concept = random.sample(concept_list,1)[0]
            negative_pairs.append((mention,concept))
    return synonym_pairs,negative_pairs

#split training and test data for one file that corresponds to the synonym_pairs
def data_split(concept_list,synonym_pairs,is_unseen=True,test_size = 0.33):
    """
    args:
    is_unseen:if is_unseen==true, then the concepts in training pairs and testing pairs will not overlap 
    returns:
    three folds of train,test datasets
    """
    datasets_folds=[]
    #notice that we collect the (mention,concept) pairs in a order of all the concepts, so the same concepts will assemble together
    #as a result, we could remove all the (mention,concept) pairs with the same concept in an easy manner 
    mentions = [mention for (mention,concept) in synonym_pairs] 
    concepts = [concept for (mention,concept) in synonym_pairs]
    
    
    #random split
    if is_unseen == False:
        for _ in range(3):
            mentions_train,mentions_test,concepts_train,concepts_test = train_test_split(
                mentions,concepts,test_size=test_size)#have already set up seed 
            
            #print(mentions_train[:3],concepts_train[:3],mentions_test[:3],concepts_test[:3])
            datasets_folds.append((mentions_train,concepts_train,mentions_test,concepts_test))
    
    #random split, and the concepts in train set and test set will not overlap
    else:
        for seed in range(3):
            mentions_train,mentions_test,concepts_train,concepts_test=mentions.copy(),[],concepts.copy(),[]
            
            left_concepts = sorted(list(set(concepts)))
            while len(mentions_test) < len(mentions) * test_size:
                concept = random.sample(left_concepts,1)[0]
                
                start_index,end_index = concepts.index(concept), len(concepts)-1 -  list(reversed(concepts)).index(concept)#the start index and the end index of the same concept

                for K in range(start_index,end_index+1):
                    mentions_test.append(mentions[K])
                    mentions_train.remove(mentions[K])
                    concepts_test.append(concept)
                    concepts_train.remove(concept)
                
                
                left_concepts.remove(concept)

            datasets_folds.append((mentions_train,concepts_train,mentions_test,concepts_test))

            #check overlap
            #for concept in concepts_test:
            #    if concept in concepts_train:
            #        print(concept)
                
    return datasets_folds


#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)        



if __name__ == '__main__':
    setup_seed(0)
    concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph('../data/datasets/cl.obo')
    datasets_folds =  data_split(concept_list=concept_list,synonym_pairs=synonym_pairs,is_unseen=True,test_size=0.33)