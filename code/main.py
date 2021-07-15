import numpy as np
import random
import os

from data_process import construct_graph,data_split

from model import EditDistance_Classifier

#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    setup_seed(0)
    dir = '../data/datasets'
    for dir,_,files in os.walk(dir):
        for filename in files:
            #if filename=='envo.obo':continue# envo.obo has a specific problem and I am not sure why.
            concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph(os.path.join(dir,filename))
            print(filename,'number of synonym pairs',len(synonym_pairs))
            if len(synonym_pairs)<10000 or len(synonym_pairs)>10000:continue#modify the two number to control how many datasets will be tested.
            datasets_folds =  data_split(concept_list=concept_list,synonym_pairs=synonym_pairs,is_unseen=True,test_size=0.33)

            for fold,data_fold in enumerate(datasets_folds):
                mentions_train,concepts_train,mentions_test,concepts_test = data_fold
                classifier = EditDistance_Classifier(concept_list)
                accu1,accu5 = classifier.eval(mentions_test,concepts_test)
                print(filename,'fold--%d,accu1--%f,accu5--%f'%(fold,accu1,accu5))

    
