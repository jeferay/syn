import numpy as np
import torch
import argparse
import logging
import time
import pdb
import os
import json
import random
from utils import (
    evaluate
)
from tqdm import tqdm
from biosyn_model import (
    CandidateDataset, 
    RerankNet, 
    BioSyn
)
from biosyn_model.data_loader import MyDataset

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Biosyn train')

    # Required
    parser.add_argument('--model_dir', required=True, 
                        help='Directory for pretrained model')
    parser.add_argument('--filedir',type=str,default='../data/datasets')

    parser.add_argument('--filename', type=str,
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output')
    
    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)

    # Train config
    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--initial_sparse_weight',
                        default=0, type=float)
    parser.add_argument('--dense_ratio', type=float,
                        default=0.5)
    parser.add_argument('--save_checkpoint_all', action="store_true")

    parser.add_argument('--device',default=0,type=int)

    parser.add_argument('--migration_rate',default=0.,type=float)#将synonym迁移到name里面作为dictionary的概率

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_loader, model):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x)  
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss
    
def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # load dictionary and queries
    mydataset = MyDataset(os.path.join(args.filedir,args.filename),migration_rate=args.migration_rate)
    train_dictionary = mydataset.load_dict_data()
    train_queries = mydataset.load_query_data(mode = 'train')
        
    # filter only names:提取出所有pair中的name，相当于他们的label是后面的cui
    names_in_train_dictionary = train_dictionary[:,0]
    #print('shape_names_in_dict%d'%len(names_in_train_dictionary))  71924
    names_in_train_queries = train_queries[:,0]
    #print('shaple_names_in_quer%d'%len(names_in_train_queries))    1587

    # load BERT tokenizer, dense_encoder, sparse_encoder
    biosyn = BioSyn()
    
    #这里是dense encoder
    encoder, tokenizer = biosyn.load_bert(
        path=args.model_dir, 
        max_length=args.max_length,
        use_cuda=args.use_cuda,
    )
    #用dictionary中的names作为corpus得到tf-idf的encoder（此时相当于train一个tf-idfencoder）
    sparse_encoder = biosyn.train_sparse_encoder(corpus=names_in_train_dictionary)#利用train_dict中的name作为所有的corpus计算tf-idf编码
    sparse_weight = biosyn.init_sparse_weight(
        initial_sparse_weight=args.initial_sparse_weight,
        use_cuda=args.use_cuda
    )
    
    # load rerank model
    model = RerankNet(
        encoder = encoder,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        sparse_weight=sparse_weight,
        use_cuda=args.use_cuda
    )
    
    #类似对比的，我们可以把dictionary当成是所有的term，query当成是所有的synonym
    # embed sparse representations for query and dictionary
    # Important! This is one time process because sparse represenation never changes.
    LOGGER.info("Sparse embedding")
    train_query_sparse_embeds = biosyn.embed_sparse(names=names_in_train_queries)
    train_dict_sparse_embeds = biosyn.embed_sparse(names=names_in_train_dictionary)
    train_sparse_score_matrix = biosyn.get_score_matrix(
        query_embeds=train_query_sparse_embeds, 
        dict_embeds=train_dict_sparse_embeds
    )
    #print(train_sparse_score_matrix.shape)1587*71924，对应于各自的len的统计结果
    train_sparse_candidate_idxs = biosyn.retrieve_candidate(
        score_matrix=train_sparse_score_matrix, 
        topk=args.topk
    )
    #print("length of candidate idxs%d"%len(train_sparse_candidate_idxs)) #1587,相当于每个query对应一组idxs
    #print("length of idxs[0]%d"%len(train_sparse_candidate_idxs[0]))# 20 符合预期top,先找到topk最后按照比例分配
    #print(train_sparse_candidate_idxs[0])

    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = tokenizer, 
        topk = args.topk, 
        d_ratio=args.dense_ratio,
        s_score_matrix=train_sparse_score_matrix,
        s_candidate_idxs=train_sparse_candidate_idxs
    )

    #print(train_set.check_label('D011125','D011125|175100')),所以只要cui由‘|’分割并包含就可以认为是label为1
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True
    )

    start = time.time()
    for epoch in range(1,args.epoch+1):
        # embed dense representations for query and dictionary for train
        # Important! This is iterative process because dense represenation changes as model is trained.
        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        train_query_dense_embeds = biosyn.embed_dense(names=names_in_train_queries, show_progress=True)
        train_dict_dense_embeds = biosyn.embed_dense(names=names_in_train_dictionary, show_progress=True)
        #print(train_dict_dense_embeds.shape) 71924 * 768
        
        train_dense_score_matrix = biosyn.get_score_matrix(
            query_embeds=train_query_dense_embeds, 
            dict_embeds=train_dict_dense_embeds
        )
        train_dense_candidate_idxs = biosyn.retrieve_candidate(
            score_matrix=train_dense_score_matrix, 
            topk=args.topk
        )
        # replace dense candidates in the train_set,getitem的时候自动利用更新好的dense candidate
        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)

        train_loss = train(args, data_loader=train_loader, model=model)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            biosyn.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == args.epoch:
            biosyn.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    
if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device('cuda:{}'.format(args.device))
    main(args)
