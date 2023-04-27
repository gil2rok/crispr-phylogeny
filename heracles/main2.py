import torch
import numpy as np
import random
import mlflow

from heracles.trainer2 import train

def main(char_matrix, mutation_rate, deletion_rate, transition_prob,
         seed, num_epochs, lr, embedding_dim, rho, stabilize, est_tree_method, true_tree=None):
    
    # validate args
    validate_args(char_matrix, mutation_rate, deletion_rate, transition_prob,
                  seed, num_epochs, lr, embedding_dim, rho, stabilize, est_tree_method, true_tree)
    
    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # set up mlflow
    mlflow.set_tracking_uri('http://127.0.0.1:5000')  # set connection
    mlflow.set_experiment('heracles training') # set the experiment
    
    # remove duplicates from char_matrix
    char_matrix = char_matrix.drop_duplicates().to_numpy(dtype=int)
    
    # compute shape of data parameters
    num_cells, num_sites = char_matrix.shape # number of cells and target sites
    num_states = len(transition_prob) # number of states
    
    # train model with mlflow logging
    with mlflow.start_run():
        mlflow.log_params({ 'mutation_rate': mutation_rate,
                           'deletion_rate': deletion_rate,
                           'transition_prob': transition_prob,
                           'num_cells': num_cells,
                           'num_sites': num_sites,
                           'num_states': num_states,
                           'seed': seed,
                           'num_epochs': num_epochs,
                           'lr': lr,
                           'embedding_dim': embedding_dim,
                           'rho': rho,
                           'stabilize': stabilize,
                           'est_tree_method': est_tree_method})
        # mlflow.log_artifact('char_matrix', char_matrix)
        # if true_tree is not None: mlflow.log_artifact('true_tree', true_tree)
        
        return train(char_matrix, mutation_rate, deletion_rate, transition_prob,
                     num_cells, num_sites, num_states,
                     seed, num_epochs, lr, embedding_dim, rho, stabilize, est_tree_method, true_tree)
        
def validate_args(char_matrix, mutation_rate, deletion_rate, transition_prob,
                  seed, num_epochs, lr, embedding_dim, rho, stabilize, est_tree_method, true_tree):
    
    try:
        rho >= 0
    except:
        raise ValueError('rho must be non-negative')