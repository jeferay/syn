"""
implement the classifier with training, evaluation,saving models and other functions
"""

import Levenshtein
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

from dataset import Biosyn_Dataset, Mention_Dataset, load_data,data_split
from evaluator import Evaluator
from models import Biosyn_Model
from criterion import marginal_loss
from transformers import *
from sklearn.feature_extraction.text import TfidfVectorizer
import os


class Biosyn_Classifier():
    def __init__(self,args):
        self.args = args
        self.filename = self.args['filename']
        self.use_text_preprocesser = self.args['use_text_preprocesser']
        self.device = self.args['device']
        self.name_array,query_id_array,self.mention2id,self.graph = load_data(self.filename,self.use_text_preprocesser)
        self.queries_train,self.queries_valid,self.queries_test = data_split(query_id_array=query_id_array,is_unseen=self.args['is_unseen'],test_size=0.33)
        self.tokenizer = BertTokenizer(vocab_file=self.args['vocab_file'])

        self.biosyn_model =Biosyn_Model(model_path = self.args['model_path'],initial_sparse_weight = self.args['initial_sparse_weight'],device=self.device )
        
        self.sparse_encoder = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
        self.sparse_encoder.fit(self.name_array)
        

    # get the embeddings of mention_array(name_array or query_array)
    def get_mention_array_bert_embedding(self,mention_array):
        
        # use dataset to help embed the mention_array
        self.biosyn_model.eval()#进入eval模式
        
        mention_dataset = Mention_Dataset(mention_array,self.tokenizer)
        mentions_embedding = []
        data_loader = DataLoader(dataset=mention_dataset,batch_size=1024)
        with torch.no_grad():
            for i,(input_ids, attention_mask) in enumerate(data_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                cls_embedding = self.biosyn_model.bert_encoder(input_ids,attention_mask).last_hidden_state[:,0,:]# batch * hidden_size
                mentions_embedding.append(cls_embedding)
            
            mentions_embedding = torch.cat(mentions_embedding, dim=0)# len(mentions) * hidden_size
            #print(mentions_embedding.shape)
        
        return mentions_embedding# still on the device

    def get_score_matrix(self,query_array):
        query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(query_array).toarray()).to(self.device)
        name_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(self.name_array).toarray()).to(self.device)
        sparse_score_matrix = torch.matmul(query_sparse_embedding,name_sparse_embedding.transpose(0,1))

        query_bert_embedding = self.get_mention_array_bert_embedding(query_array).to(self.device)
        name_bert_embedding = self.get_mention_array_bert_embedding(self.name_array).to(self.device)
        bert_score_matrix = torch.matmul(query_bert_embedding,name_bert_embedding.transpose(0,1))

        return sparse_score_matrix,bert_score_matrix


    def train(self):
        print('in train')
        criterion = marginal_loss
        optimizer = torch.optim.Adam([
            {'params': self.biosyn_model.bert_encoder.parameters()},
            {'params': self.biosyn_model.sparse_weight, 'lr': 0.01, 'weight_decay': 0}
            ], 
            lr=self.args['lr'], weight_decay=self.args['weight_decay']
        )
        
        for epoch in range(1, self.args['epoch_num']):
            loss_sum = 0
            self.biosyn_model.train()

            sparse_score_matrix,bert_score_matrix = self.get_score_matrix(self.queries_train)
            biosyn_dataset = Biosyn_Dataset(self.name_array,self.queries_train,self.mention2id,self.args['top_k'],sparse_score_matrix,bert_score_matrix,self.args['bert_ratio'],self.tokenizer)
            data_loader = DataLoader(dataset=biosyn_dataset,batch_size=self.args['batch_size'])
            for iteration,batch_data in tqdm(enumerate(data_loader),total=len(data_loader)):

                optimizer.zero_grad()

                query_ids,query_attention_mask,candidates_names_ids,candidates_names_attention_mask,candidates_sparse_score,labels = batch_data
                query_ids = query_ids.to(self.device)
                query_attention_mask = query_attention_mask.to(self.device)
                candidates_names_ids = candidates_names_ids.to(self.device)
                candidates_names_attention_mask = candidates_names_attention_mask.to(self.device)
                candidates_sparse_score = candidates_sparse_score.to(self.device)
                labels = labels.to(self.device)
                score = self.biosyn_model.forward(query_ids,query_attention_mask,candidates_names_ids,candidates_names_attention_mask,candidates_sparse_score)
                
                loss = criterion(score,labels)
                loss_sum+=loss.item()
                loss.backward()
                optimizer.step()
            
            loss_sum/=len(self.queries_train)

            
            
            if self.args['save_checkpoint_all'] or epoch == self.args['epoch_num']:
                checkpoint_dir = os.path.join(self.args['exp_path'], "checkpoint_{}".format(epoch))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                self.save_model(checkpoint_dir)
            
            accu1,accu_k = self.eval(self.queries_valid)
            self.args['logger'].info("epoch %d done, accu_1 = %.2f, accu_%d = %f"%(epoch,float(accu1),self.args['eval_k'], float(accu_k)))

            


    def eval(self,query_array,load_model=False):
        self.biosyn_model.eval()
        sparse_score_matrix,bert_score_matrix = self.get_score_matrix(query_array)
        if self.args['score_mode'] == 'hybrid':
            score_matrix = self.biosyn_model.sparse_weight * sparse_score_matrix + bert_score_matrix
        elif self.args['score_mode'] == 'sparse':
            score_matrix = sparse_score_matrix
        else:
            score_matrix = bert_score_matrix
        sorted,indices = torch.sort(score_matrix,descending=True)# 降序，重要
        labels = torch.LongTensor([self.mention2id[query] for query in query_array]).to(self.args['device'])
        accu_1 = (indices[:,0]==labels).sum()/len(query_array)
        accu_k = (indices[:,:self.args['eval_k']]== torch.unsqueeze(labels,dim=1)).sum()/len(query_array)

        return accu_1,accu_k
        



    
    def save_model(self,checkpoint_dir):
        self.biosyn_model.bert_encoder.save_pretrained(checkpoint_dir)
        torch.save(self.biosyn_model.sparse_weight,os.path.join(checkpoint_dir,'sparse_weight.pth'))

    def load_model(self,model_path):
        self.args['logger'].info('model loaded at %s'%model_path)
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.biosyn_model.bert_encoder.load_state_dict(state_dict,False)
        self.biosyn_model.sparse_weight = torch.load(os.path.join(model_path,'sparse_weight.pth'))
        

#edit distance model, calulate the similarity accroding edit distance
class EditDistance_Classifier():
    def __init__(self,concepts_list) :
        self.concepts_list = concepts_list
    
    def softmax(self,array):
        exp_res = np.exp(array)
        sum_exp = np.sum(exp_res)
        return exp_res/sum_exp

    #use mean_distance/distance as the similarity score
    def forward(self,mentions):
        """
        return: the score arrays of all concepts over all samples
        """
        score_matrix=[]
        for j,mention in tqdm(enumerate(mentions)):
            distance_array=np.ones(len(self.concepts_list))
            for i,concept in enumerate(self.concepts_list):
                distance_array[i] = Levenshtein.distance(mention,concept) + 1#in case that edit distance equals to zero
            score_array =  self.softmax(np.divide(distance_array.mean(),distance_array))
            score_matrix.append(score_array)
        score_matrix = np.stack(score_matrix,axis=0)#shape:n_samples * n_concepts
        #print(score_matrix.shape)
        
        return score_matrix
    
    def eval(self,mentions,concepts):
        """
        inputss: mentions and concepts are both list of string 
        """
        score_matrix = self.forward(mentions)
        true_labels = np.array([self.concepts_list.index(concept) for concept in concepts])
        evaluator = Evaluator()
        accu1 = evaluator.accu(score_matrix,true_labels,top_k=1)
        accu5 = evaluator.accu(score_matrix,true_labels,top_k=5)
        return accu1,accu5
        









        

            






