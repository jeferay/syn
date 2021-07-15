import numpy as np
import random

from data_process import construct_graph,data_split

from model import EditDistance_Classifier

#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    setup_seed(0)
    concept_list,concept2id,edges,mention_list,synonym_pairs = construct_graph()
    datasets_folds =  data_split(concept_list=concept_list,synonym_pairs=synonym_pairs,is_unseen=True,test_size=0.33)

    mentions_train,concepts_train,mentions_test,concepts_test = datasets_folds[0]

    classifier = EditDistance_Classifier(concept_list)
    accu1,accu5 = classifier.eval(mentions_test,concepts_test)
    print(accu1,accu5)
    
