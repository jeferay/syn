import numpy as np
import random
import os
import torch
import logging
import argparse
from classifier import  Biosyn_Classifier, Graphsage_Classifier




#set up seed         
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

# set up logger
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    if not os.path.exists(name):
        os.makedirs(name)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename',type=str,default='../data/datasets/cl.obo')
    parser.add_argument('--use_text_preprocesser',action='store_true',default=False)
    parser.add_argument('--is_unseen',action='store_true',default=False)
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--model_path',type=str,default='../exp/eco/checkpoint_9')
    parser.add_argument('--vocab_file',type=str,default='../biobert/vocab.txt')
    parser.add_argument('--initial_sparse_weight',type=float,default=1.)
    parser.add_argument('--bert_ratio',type=float,default=0.5)
    parser.add_argument('--bert_lr',type=float,default=1e-5)
    parser.add_argument('--bert_weight_decay',type=float,default=1e-3)
    parser.add_argument('--graph_lr',type=float,default=0.01)
    parser.add_argument('--graph_weight_decay',type=float,default=5e-5)

    parser.add_argument('--epoch_num',type=int,default=10)
    parser.add_argument('--top_k',type=int,default=20)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--exp_path',type=str,default='../exp/cl')
    parser.add_argument('--score_mode',type=str,default='hybrid')
    parser.add_argument('--eval_k',type=int,default=20)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--save_checkpoint_all',action='store_true',default=True)

    
    args = parser.parse_args()
    args = args.__dict__
    
    torch.cuda.set_device(args['device'])
    args['device'] = torch.device(args['device'] if torch.cuda.is_available() else 'cpu')

    logger = setup_logger(name=args['exp_path'][:],log_file=os.path.join(args['exp_path'],'log.log'))
    args['logger'] = logger

    setup_seed(args['seed'])

    bio = Graphsage_Classifier(args)
    bio.train()
    bio.eval(bio.queries_train)

