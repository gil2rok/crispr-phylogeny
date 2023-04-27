import argparse
import mlflow
import pickle
import torch
import numpy as np
import random

from trainer import train
from os import path 

def main(args):
    # validate and transform arguments
    args = validate_args(args)
    args.rho = torch.tensor(args.rho, dtype=torch.float64)
    
    # set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # load and process data
    true_tree, params = load_data(args)
    char_matrix = true_tree.character_matrix.drop_duplicates().to_numpy(dtype=int) # character matrix of leaf cells

    deletion_rate = params['deletion_rate'] # global deletion rate
    mutation_rate = params['mutation_rate'] # site-specific mutation rate
    indel_distribution = list(params['transition_prob'].values())
    
    num_cells = char_matrix.shape[0] # number of cells
    num_sites = char_matrix.shape[1] # number of target sites
    num_states = len(params['transition_prob']) # number of states
    
    # set up mlflow
    mlflow.set_tracking_uri('http://127.0.0.1:5000')  # set connection
    mlflow.set_experiment(args.exp_name) # set the experiment
    run_name = None # TODO
    
    args_dict = dict([(a, getattr(args, a)) for a in dir(args) if not a.startswith('__')]) # convert args to dict for logging
    args_dict['rho'] = args.rho.item() # convert rho to float for logging
    
    # train model with mlflow logging
    with mlflow.start_run(): # TODO: add run name
        mlflow.log_params(args_dict)
        mlflow.log_params({'deletion_rate': deletion_rate,
                          'mutation_rate': mutation_rate,
                          'indel_distribution': indel_distribution,
                          'num_cells': num_cells,
                          'num_sites': num_sites,
                          'num_states': num_states})
        
        train(char_matrix, deletion_rate, mutation_rate, indel_distribution, 
              num_cells, num_sites, num_states, args, true_tree)
    
def load_data(args):
    """ load data
    
    Args:
        args (args): parsed arguments
    """
    
    # load true tree
    with open(args.tree_path, 'rb') as file:
        true_tree = pickle.load(file)
    
    # load params
    with open(args.params_path, 'rb') as file:
        params = pickle.load(file)
    
    return true_tree, params
    
def validate_args(args):
    """ validate and transform arguments

    Args:
        args (args): parsed arguments
    """
    
    try:
        args.rho >= 0
    except:
        raise ValueError('rho must be non-negative')
    
    try:
        path.exists(args.tree_path)
    except:
        raise ValueError('tree_path does not exist')
    
    try:
        path.exists(args.params_path)
    except:
        raise ValueError('params_path does not exist')
    
    try:
        path.exists(args.save_path)
    except:
        raise ValueError('save_path does not exist')
    
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='heracles',
                                     description='reconstruct lineage of CRISPR-edited cells with hyperbolic embeddings'
                                    )
    # general arguments
    general = parser.add_argument_group('general arguments')
    general.add_argument('--tree_path', type=str, default='data/true_tree',
                         help="path to picked Cassiopeia tree")
    general.add_argument('--params_path', type=str, default='data/params',
                         help='path to picked params dict')
    general.add_argument('--save_path', type=str, default='saved',
                         help='path to save best hyperbolic embeddings')
    general.add_argument('--seed', default=0, type=int,
                         help='random seed')
    general.add_argument('--exp_name', default='heracles training', type=str,
                         help='experiment name for mlflow logging')
    
    # training arguments
    training = parser.add_argument_group('training arguments')
    training.add_argument('--num_epochs', default=30, type=int,
                          help='number of epochs to train')
    training.add_argument('--lr', default=5e-2, type=float,
                          help='learning rate')
    training.add_argument('--embedding_dim', default=3, type=int,
                              help='dimension of hyperbolic embeddings')
    training.add_argument('--rho', default=2, type=float,
                          help='negative curvature of hyperbolic space')
    training.add_argument('--stabilize', default=1, type=int,
                          help='project embeddings back to manifold')
    training.add_argument('--tree_reconstruction', default='neighbor-joining', type=str,
                          choices=['upgma', 'neighbor-joining'],
                          help='method to reconstruct tree from pairwise distances')
    
    # parse arguments
    args = parser.parse_args()
    main(args)