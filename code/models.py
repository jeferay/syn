import torch
import torch.nn as nn
from transformers import *
import os
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class Biosyn_Model(nn.Module):
    def __init__(self,model_path,initial_sparse_weight,device):
        super(Biosyn_Model,self).__init__()
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.to(device)
        self.sparse_weight = nn.Parameter(torch.empty(1).cuda(0))
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
    def __init__(self,feature_size,hidden_size,output_size,model_path,initial_sparse_weight,device):
        super(Graphsage_Model,self).__init__()

        #load bert encoder
        config = BertConfig.from_json_file(os.path.join(model_path, "config.json"))
        self.bert_encoder = BertModel(config = config) # bert encoder
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        self.bert_encoder.load_state_dict(state_dict,False)
        self.bert_encoder = self.bert_encoder.to(device)

        self.sparse_weight = nn.Parameter(torch.empty(1).cuda())
        self.sparse_weight.data.fill_(initial_sparse_weight)

        self.sage1 = SAGEConv(feature_size,hidden_size).to(device)
        self.sage2 = SAGEConv(hidden_size,output_size).to(device)

    # decide the candidates set in forward set;here we simply decide the candidates by the sum of sparse scores and dense scores
    # in order to choose enouth positive samples, we put the positive samples into candidates set on purpose
    def forward(self,query_ids,query_attention_mask,sparse_score,names_bert_embedding,query_indices,edge_index,top_k,is_training=True):
        query_bert_embedding = self.bert_encoder(query_ids,query_attention_mask).last_hidden_state[:,0,:]# shape of (batch,hidden_size)
        names_graph_embedding = self.sage1(names_bert_embedding,edge_index)
        names_graph_embedding = F.relu(names_graph_embedding)
        names_graph_embedding = F.dropout(names_graph_embedding)
        names_graph_embedding = self.sage2(names_graph_embedding,edge_index)# shape of (N, hidden_size)
        dense_score = torch.matmul(query_bert_embedding, torch.transpose(names_graph_embedding,dim0=0,dim1=1))# shape of (batch, N)4
        score = dense_score +  sparse_score * self.sparse_weight
        sorted_score,indices = torch.sort(score,descending=True)# descending. important, tensors of shape(batch, N)
        # when training, inject the positive samples
        if is_training:
            posi_score = torch.diag(score[:,query_indices])# tensor of shape(batch,)
            sorted_score = torch.cat([posi_score.unsqueeze(dim=1),sorted_score], dim=1)
            indices = torch.cat([query_indices.unsqueeze(dim=1),indices], dim=1)# inject the positive samples on purpose

        return sorted_score[:,:top_k],indices[:,:top_k]# return the top k scores and indices, including the positive samples









class SimpleEmbedding(nn.Module):
    def __init__(self):
        super(SimpleEmbedding, self).__init__()
        self.layer = nn.Sequential(nn.Conv1d(1, 1, kernel_size=200+1-128), nn.PReLU(), nn.MaxPool1d(3, stride=1, padding=1))

    def forward(self, x):
        x = self.layer(x)
        return x


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)








