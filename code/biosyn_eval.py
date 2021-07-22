import argparse
import logging
import os
import json
from tqdm import tqdm
from utils import (
    evaluate
)
from biosyn_model import (
    BioSyn
)

from biosyn_model.data_loader import MyDataset
LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='BioSyn evaluation')

    # Required
    parser.add_argument('--model_dir', required=True, help='Directory for model')
    parser.add_argument('--filename', type=str, required=True, help='data set to evaluate')

    # Run settings
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--topk',  type=int, default=20)
    parser.add_argument('--score_mode',  type=str, default='hybrid', help='hybrid/dense/sparse')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--save_predictions', action="store_true", help="whether to save predictions")

    # Tokenizer settings
    parser.add_argument('--max_length', default=25, type=int)
    
    args = parser.parse_args()
    return args
    
def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def main(args):
    init_logging()
    print(args)

    # load dictionary and queries
    mydataset = MyDataset(args.filename)
    eval_dictionary = mydataset.load_dict_data()
    eval_queries = mydataset.load_query_data(mode = 'eval')
    biosyn = BioSyn().load_model(
            path=args.model_dir,
            max_length=args.max_length,
            use_cuda=args.use_cuda
    )
    
    result_evalset = evaluate(
        biosyn=biosyn,
        eval_dictionary=eval_dictionary,
        eval_queries=eval_queries,
        topk=args.topk,
        score_mode=args.score_mode
    )
    
    LOGGER.info("acc@1={}".format(result_evalset['acc1']))
    LOGGER.info("acc@5={}".format(result_evalset['acc5']))
    
    if args.save_predictions:
        output_file = os.path.join(args.output_dir,"predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)
