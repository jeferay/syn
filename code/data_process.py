
import json
import wget
import os
import scipy
import  tqdm
import random
#catch all suitable datasets on the website
def get_all_data(filename='../data/ontologies.jsonld'):
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
                    if product['id']==id + '.obo' and id!='rs' and id!='fix' and id!='eo':# for some unkonwn reasons, rs.obo, fix.obo and eo.obo can not be downloaded
                        urls.append(product['ontology_purl'])
                        ids.append(id)
    
    #download relative files to data_dir, finnally we get 95 files
    #print(ids)
    data_dir = '../data/datasets'
    for i,(id,url) in  enumerate(zip(ids,urls)):
        #print(id)
        filename = id+'.obo'
        file = wget.download(url=url,out= os.path.join(data_dir,filename))
    
    
#generate relative concepts ,construct a scipy csr matrix of edges to present the graph and collect synonym pairs
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
        node = ""
        iter_concept = iter(concept_list)
        for i,line in enumerate(lines):
            if line[:6]=='[Term]':#starts with a [Term] and ends with an '\n'
                check_new_term = True
                node = next(iter_concept)
                continue
            if line[:1]=='\n':
                check_new_term = False
                continue
            if check_new_term == True:
                if line[:5]=='is_a:':
                    entry = line.split(" ")
                    father_node = " ".join(entry[entry.index('!') + 1:])[:-1]
                    edges.append((concept2id[father_node],concept2id[node]))
                    #print("--node--",node,'---father_node---',father_node)
                if line[:8]=='synonym:':
                    start_pos = line.index("\"") + 1
                    end_pos = line[start_pos:].index("\"") + start_pos
                    mention = line[start_pos:end_pos]
                    mention_set.add(mention)
                    synonym_pairs.append((mention,node))
                    #print("--node--",node,'---synonym---',synonym)
        
        return concept_list,concept2id,edges,list(mention_set),synonym_pairs

#generate negative samples whose number is equal to the number of positive ones
def construct_positive_and_negative_pairs(concept_list,synonym_pairs):
    """
    returns: positive pairs and negative pairs
    """
    negative_pairs = []
    random.seed(0)
    for i,(mention,_) in enumerate(synonym_pairs):
        concept = random.sample(concept_list,1)[0]
        while (mention,concept) in synonym_pairs:
            concept = random.sample(concept_list,1)[0]
        negative_pairs.append((mention,concept))
    return synonym_pairs,negative_pairs

         

if __name__ == '__main__':
    concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph()
    print(edges[0])
    a,b = edges[0]
    print(concept_list[a],concept_list[b])
    print(len(concept_list))
    synonym_pairs,negative_pairs = construct_positive_and_negative_pairs(concept_list,synonym_pairs)
    print(len(negative_pairs),len(synonym_pairs))
    print(negative_pairs[0],synonym_pairs[0])