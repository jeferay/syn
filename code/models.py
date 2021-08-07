from numpy.core.numeric import indices
import torch
import torch.nn as nn
from transformers import *
import os
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class Biosyn_Model(nn.Module):
    def __init__(self,model_path,initial_sparse_weight):
        super(Biosyn_Model,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.cuda()
        self.sparse_weight = nn.Parameter(torch.empty(1)).cuda()
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
        self.sparse_weight = nn.Parameter(torch.empty(1)).cuda()
        self.sparse_weight.data.fill_(1.0)

        self.load_model(model_path=model_path)# candidates model have been trained already

        self.sage1 = GCNConv(feature_size,hidden_size).cuda()
        self.sage2 = GCNConv(hidden_size,output_size).cuda()

        self.score_network = nn.Linear(in_features=output_size + feature_size,out_features=1).cuda()# input size will be the sum of query size and entity size

    def load_model(self,model_path):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.sparse_weight.weight = torch.load(os.path.join(model_path,'sparse_weight.pth'))

    def forward(self,query_ids,query_attention_mask,query,names_embedding,edge_index,candidates_indices,top_k):
        """
            args:candidates_indices: shape(batch, top_k)
        """
        query_sparse_embedding = torch.FloatTensor(self.sparse_encoder.transform(query).toarray()).cuda()
        query_bert_embedding =  self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        query_embedding = torch.cat((query_sparse_embedding,query_bert_embedding), dim =1)# tensor of shape(batch,featur_size)
        names_graph_embedding = self.sage1(names_embedding,edge_index)
        names_graph_embedding = torch.sigmoid(F.dropout(names_graph_embedding,0.3))
        names_graph_embedding = self.sage2(names_graph_embedding,edge_index)# shape of (N, output_size)

        score = []
        for i in range(top_k):
            ith_indices = candidates_indices[:,i]# the ith index for every query, tensor of shape(batch,)
            ith_candidate_graph_embedding = names_graph_embedding[ith_indices]# tensor of shape(batch,hidden)
            ith_score_embedding = torch.cat((ith_candidate_graph_embedding,query_embedding),dim=1)#tensor of shape(batch,feature_size + output_size)
            ith_score = self.score_network(ith_score_embedding)#tensor of shape(batch,1)
            score.append(ith_score)
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
        self.sparse_weight = nn.Parameter(torch.empty(1)).cuda()
        self.sparse_weight.data.fill_(initial_sparse_weight)
    
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
        query_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        candidiate_names_bert_embedding = []
        for i in range(candidates_ids.shape[1]):#top_k
            ids = candidates_ids[:,i,:]
            attention_mask = candidates_attention_mask[:,i,:]
            cls_embedding = self.bert_encoder(ids,attention_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            candidiate_names_bert_embedding.append(cls_embedding)
        candidiate_names_bert_embedding = torch.stack(candidiate_names_bert_embedding,dim = 1)# tensor of shape(batch, top_k, hidden_size)

        query_embedding = torch.unsqueeze(query_embedding,dim=1)#batch * 1 *hidden_size
        bert_score = torch.bmm(query_embedding, candidiate_names_bert_embedding.transpose(dim0=1,dim1=2)).squeeze()# batch * top_k

        score = bert_score + candidates_sparse_score * self.sparse_weight
        return score


        
class Bert_Cross_Encoder(nn.Module):
    def __init__(self,model_path):
        super(Bert_Cross_Encoder,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.cuda()
        self.linear = nn.Linear(in_features=768,out_features=768).cuda()


    
    def load_model(self,model_path):
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.linear.load_state_dict(torch.load(os.path.join(model_path,'linear.pth')))

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
    
    def forward(self,query_ids,query_attention_mask,candidates_ids,candidates_attention_mask):
        query_embedding =  self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]
        #query_embedding = F.dropout(query_embedding,0.3)
        candidiate_names_bert_embedding = []
        for i in range(candidates_ids.shape[1]):#top_k
            ids = candidates_ids[:,i,:]
            attention_mask = candidates_attention_mask[:,i,:]
            cls_embedding = self.bert_encoder(ids,attention_mask).last_hidden_state[:,0,:]#tensor of shape(batch, hidden_size)
            candidiate_names_bert_embedding.append(cls_embedding)
        candidiate_names_bert_embedding = torch.stack(candidiate_names_bert_embedding,dim = 1)# tensor of shape(batch, top_k, hidden_size)

        #candidiate_names_bert_embedding = F.dropout(candidiate_names_bert_embedding,0.3)
        top_k = candidates_ids.shape[1]
        score = []
        for k in range(top_k):
            k_names_embedding = candidiate_names_bert_embedding[:,k,:]# (batch, hidden_size)
            k_linear = self.linear(query_embedding)# tensor of shape(batch,hidden_size)
            k_names_embedding = torch.unsqueeze(k_names_embedding,dim=2)
            k_linear = torch.unsqueeze(k_linear,dim=1)
            k_score = torch.bmm(k_linear,k_names_embedding).squeeze(2)# tensor of shape(batch,1)
            score.append(k_score)
        score = torch.cat(score,dim=1)# tensor of shape(batch,top_k)
        return score











