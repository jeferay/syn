import os
import sys
sys.path.insert(0,"/home/megh/projects/entity-norm/syn/")
from code.dataset import *
import numpy as np
import torch

class Sieves():

    def __init__(self, dataset_path = os.path.join("../../data/datasets/cl.obo")) -> None:
        self.name_array, self.query_id_array, self.mention2id, self.edge_index  = load_data(filename=dataset_path)
        self.name2id, self.query2id  = self.process_data_for_seive()
        self.normalized_queries = {}
        self.current_unnormalized_queries = self.query2id.copy()
        self.unnormalized_queries_final  = {}


    def process_data_for_seive(self):
        name2id = {}
        for name_ in self.name_array:
            id_number =  self.mention2id[str(name_)]
            name2id[str(name_)] = id_number
        query2id  = {}
        for item in self.query_id_array:
            query2id[str(item[0])] =  int(item[1])
        return name2id, query2id


    def exact_match(self):
        for query_ in self.current_unnomralized_queries.keys():
            for name_ in self.name_array:
                if query_ == name_:
                    self.normalized_queries[query_] = self.current_unnormalized_queries[query_].copy()
            # deleting the entries from the unnormalized queries
            for key_ in self.normalized_queries:
                del self.current_unnormalized_queries[key_]
            return 

    def concat_abbrevations_files(self, file_path_list = [os.path.join("../text_resources/semeval-wiki-abbreviations.txt"), os.path.join("../text_resources/ncbi-wiki-abbreviations.txt")]):
        all_abbrevations_dict  = {}
        for file_path in file_path_list:
            file_handle  = open(file_path,"r")
            for line in file_handle:
                line  = line.replace("\n","")
                line_split  = line.split("||")
                all_abbrevations_dict[line_split[0]] = line_split[1]

        return all_abbrevations_dict


    def abbrevation_expansion(self):

        self.all_abbrevations_dict  = self.concat_abbrevations_files()
   
        for query_ in self.current_unnormalized_queries.keys():
            # splitting the words in a query 
            words_query = query_.split
            

if __name__ == '__main__':
    s_classifier = Sieves()
    