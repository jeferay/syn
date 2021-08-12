from numpy.core.numeric import indices
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm1d
from transformers import *
import os
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.preprocessing import StandardScaler

from utils import TimeIt

class Biosyn_Model(nn.Module):
    def __init__(self,model_path,initial_sparse_weight):
        super(Biosyn_Model,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.cuda()
        self.sparse_weight = nn.Parameter(torch.empty(1).cuda(),requires_grad=True)
        self.sparse_weight.data.fill_(initial_sparse_weight)
        
    
    def forward(self,query_ids,query_attention_mask,candidates_names_ids,candidates_names_attention_mask,candidates_sparse_score):
        """
        args:
            candidates_names_ids: batch * top_k * max_len
            candidates_names_attention_mask: batch * top_k * max_len
            candidates_sparse_score: batch * top_k
        """
        query_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        candidiate_names_graph_embedding = []
        for i in range(candidates_names_ids.shape[1]):
            ids = candidates_names_ids[:,i,:]
            attention_mask = candidates_names_attention_mask[:,i,:]
            cls_embedding = self.bert_encoder(ids,attention_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            candidiate_names_graph_embedding.append(cls_embedding)
        candidiate_names_graph_embedding = torch.stack(candidiate_names_graph_embedding,dim = 1)# tensor of shape(batch, top_k, hidden_size)

        query_embedding = torch.unsqueeze(query_embedding,dim=1)#batch * 1 *hidden_size
        bert_score = torch.bmm(query_embedding, candidiate_names_graph_embedding.transpose(dim0=1,dim1=2)).squeeze()# batch * top_k

        score = bert_score + candidates_sparse_score * self.sparse_weight
        return score

class Graphsage_Model(torch.nn.Module):
    def __init__(self,feature_size,hidden_size,output_size,model_path,sparse_encoder):
        super(Graphsage_Model,self).__init__()
        
        #load bert encoder to generate candidates
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        self.bert_encoder = self.bert_encoder.cuda()
        self.sparse_encoder = sparse_encoder
        self.sparse_weight = nn.Parameter(torch.empty(1).cuda(),requires_grad=True)
        self.sparse_weight.data.fill_(1.0)

        self.load_model(model_path=model_path)# candidates model have been trained already

        self.sage1 = GCNConv(feature_size,hidden_size).cuda()
        self.sage2 = GCNConv(hidden_size,output_size).cuda()

        self.score_network = nn.Sequential(
            nn.Linear(in_features=feature_size,out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.Sigmoid(),
            nn.Linear(in_features=256,out_features=output_size)
        ).cuda()

        ########################
        list(self.score_network.modules())[0][0].weight.data.normal_(0, 1e-3)
        list(self.score_network.modules())[0][-1].weight.data.normal_(0, 1e-3)
        ########################

        
       

    def load_pretrained_model(self,model_path):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.sparse_weight.weight = torch.load(os.path.join(model_path,'sparse_weight.pth'))

    def forward(self,query_ids,query_attention_mask,query,names_embedding,edge_index,candidates_indices,top_k):
        """
            args:candidates_indices: shape(batch, top_k)
        """
        query_bert_embedding =  self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(query).toarray()).cuda()
        
        query_embedding = torch.cat((query_bert_embedding,query_sparse_embedding), dim =1)# tensor of shape(batch,bert_size + sparse_size)
        names_graph_embedding = self.sage1(names_embedding,edge_index)
        names_graph_embedding = torch.sigmoid(F.dropout(names_graph_embedding,0.3))
        names_graph_embedding = self.sage2(names_graph_embedding,edge_index)# shape of (N, output_size)

        
        score = []
        for k in range(top_k):
            k_indices = candidates_indices[:,k]# the ith index for every query, tensor of shape(batch,)
            k_candidate_graph_embedding = names_embedding[k_indices]# tensor of shape(batch,hidden)# modify
            k_candidate_graph_embedding= torch.unsqueeze(k_candidate_graph_embedding,dim=2)# tensor of shape(batch.hidden,1)
        
            k_linear = self.score_network(query_embedding)
            k_linear = torch.unsqueeze(k_linear,dim=1)
            
            k_score = torch.bmm(k_linear,k_candidate_graph_embedding).squeeze(2)# tensor of shape(batch,1)k_score = self.score_network(k_linear,k_candidate_graph_embedding)#tensor of shape(batch,1)
            score.append(k_score)
        score = torch.cat(score,dim=1)#tensor fo shape(batch,top_k)# rememebr that the first index is not the predicted result

        return score


class Bert_Candidate_Generator(nn.Module):
    def __init__(self,model_path,initial_sparse_weight):
        super(Bert_Candidate_Generator,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.cuda()
        if initial_sparse_weight == 0:
            self.sparse_weight = 0
        else:
            self.sparse_weight = nn.Parameter(torch.empty(1).cuda(),requires_grad=True)
            self.sparse_weight.data.fill_(initial_sparse_weight)

        self._cls_bn = nn.BatchNorm1d(num_features=1).cuda()

    def cls_bn(self, x):
        return self._cls_bn(x.unsqueeze(1)).squeeze(1)
    
    # load a fine tuned model
    def load_model(self,model_path):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.sparse_weight.weight = torch.load(os.path.join(model_path,'sparse_weight.pth'))

    def forward(self,query_ids,query_attention_mask,candidates_ids,candidates_attention_mask,candidates_sparse_score):
        """
        args:
            candidates_names_ids: batch * top_k * max_len
            candidates_names_attention_mask: batch * top_k * max_len
            candidates_sparse_score: batch * top_k
        """
        # print('query_ids', query_ids)
        # print('query_attention_mask', query_attention_mask)

        query_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        candidiate_names_bert_embedding = []
        for i in range(candidates_ids.shape[1]):#top_k
            ids = candidates_ids[:,i,:]
            attention_mask = candidates_attention_mask[:,i,:]
            cls_embedding = self.bert_encoder(ids,attention_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            candidiate_names_bert_embedding.append(cls_embedding)
        candidiate_names_bert_embedding = torch.stack(candidiate_names_bert_embedding,dim = 1)# tensor of shape(batch, top_k, hidden_size)

        query_embedding = torch.unsqueeze(query_embedding,dim=1)#batch * 1 *hidden_size
        bert_score = torch.bmm(query_embedding, candidiate_names_bert_embedding.transpose(dim0=1,dim1=2)).squeeze(1)# batch * top_k

        # print('query embedding', query_embedding)
        # print('candidiate_names_graph_embedding', candidiate_names_bert_embedding)

        bert_score = self.cls_bn(bert_score)

        score = bert_score + candidates_sparse_score * self.sparse_weight
        return score

        
class Bert_Cross_Encoder(nn.Module):
    def __init__(self,model_path,feature_size,sparse_encoder):
        super(Bert_Cross_Encoder,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.cuda()
        self.score_network = nn.Sequential(
            nn.Linear(in_features=feature_size,out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.Sigmoid(),
            nn.Linear(in_features=256,out_features=feature_size)
        ).cuda()
        
        ########################
        list(self.score_network.modules())[0][0].weight.data.normal_(0, 1e-3)
        list(self.score_network.modules())[0][-1].weight.data.normal_(0, 1e-3)
        ########################
        
        self.sparse_encoder = sparse_encoder


    def load_pretrained_model(self,model_path):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)

    # return corss encoder scores among all candidates(tensor of shape(batch,top_k))
    def _forward(self,pair_ids,pair_attn_mask):
        """
        args:
            pair_ids: tensor of shape(batch,top_k,max_len)
            pair_attn_mask: tensor of shape(batch,top_k,max_len)
        """
        score = []
        top_k = pair_ids.shape[1]
        for k in range(top_k):
            ids = pair_ids[:,k,:]
            attn_mask = pair_attn_mask[:,k,:]
            cls_embedding = self.bert_encoder(ids,attn_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            #cls_embedding = F.dropout(input=cls_embedding,p=0.5)
            score_k = torch.sigmoid(self.linear(cls_embedding))# tensor of shape(batch,1)
            score.append(score_k)
        score = torch.cat(score,dim=1)#tensor of shape(batch,top_k)
        return score
    

    def forward(self,query_ids,query_attention_mask,candidates_ids,candidates_attention_mask,query,names_sparse_embedding,candidates_indices):
        # print('query_ids', query_ids)
        # print('query_attention_mask', query_attention_mask)

        query_bert_embedding =  self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(query).toarray()).cuda()
        query_embedding = torch.cat((query_bert_embedding,query_sparse_embedding), dim =1)# tensor of shape(batch,bert_size + sparse_size)

        #query_embedding = F.dropout(query_embedding,0.3)
        candidiate_names_embedding = []
        for k in range(candidates_ids.shape[1]):#top_k
            k_ids = candidates_ids[:,k,:]
            k_attention_mask = candidates_attention_mask[:,k,:]
            k_bert_embedding = self.bert_encoder(k_ids,k_attention_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            k_candidates_indices = candidates_indices[:,k]
            k_sparse_embedding = names_sparse_embedding[k_candidates_indices]
            k_embedding = torch.cat((k_bert_embedding,k_sparse_embedding),dim=1)# tensor of shape(batch, 768+sparse_feature_size)


            candidiate_names_embedding.append(k_embedding)
        candidiate_names_embedding = torch.stack(candidiate_names_embedding,dim = 1)# tensor of shape(batch, top_k, hidden_size)
        # print('query embedding', query_embedding)
        # print('candidiate_names_graph_embedding', candidiate_names_graph_embedding)

        #candidiate_names_bert_embedding = F.dropout(candidiate_names_bert_embedding,0.3)
        top_k = candidates_ids.shape[1]
        score = []
        for k in range(top_k):
            k_names_embedding = candidiate_names_embedding[:,k,:]# (batch, hidden_size)
            k_linear = self.score_network(query_embedding)# tensor of shape(batch,hidden_size)
            k_names_embedding = torch.unsqueeze(k_names_embedding,dim=2)
            k_linear = torch.unsqueeze(k_linear,dim=1)
            k_score = torch.bmm(k_linear,k_names_embedding).squeeze(2)# tensor of shape(batch,1)
            score.append(k_score)
        score = torch.cat(score,dim=1)# tensor of shape(batch,top_k)
        return score










