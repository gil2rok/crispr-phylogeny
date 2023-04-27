import torch
import numpy as np
from os import path

from geoopt.optim import RiemannianSGD
from .logalike import Logalike
from mlflow import log_metric, log_artifact
from dendropy import PhylogeneticDistanceMatrix as phylo_dist

from .hyperboloid_wilson import Hyperboloid
from .metrics import cas_triplets_correct, dist_correlation
from .logalike_utils import ancestry_aware_hamming_dist

def init(char_matrix, deletion_rate, mutation_rate, indel_distribution, 
              num_cells, num_sites, num_states, args):
    
    # reconstruct infinitesimal generator Q
    Q_list = [None] * num_sites
    for i in range(num_sites):
        Q_list[i] = generate_Q(num_sites, num_states, deletion_rate, 
                               mutation_rate, indel_distribution)
        
    # intial hyperbolic embedding
    dist_matrix = char_to_dist(char_matrix) # pairwise distancess from character matrix
    est_tree = estimate_tree(dist_matrix, method=args.tree_reconstruction) # estimate phylogenetic tree
    X = embed_tree(est_tree, torch.sqrt(args.rho), num_cells, local_dim=args.embedding_dim-1)  # embed tree into hyperbolic space

    # initalize logalike object and optimizer
    l = Logalike(X, Q_list, char_matrix, num_states, args.rho, priors=None) # TODO: add priors
    opt = RiemannianSGD([l.X], lr=args.lr, stabilize=args.stabilize)
    
    return l, opt
    
def train(char_matrix, deletion_rate, mutation_rate, indel_distribution, 
              num_cells, num_sites, num_states, args, true_tree=None):
        
    # initialize logalike object and optimizer
    l, opt = init(char_matrix, deletion_rate, mutation_rate, indel_distribution, 
              num_cells, num_sites, num_states, args)
    
    # iterate over epochs
    best_epoch_loss = np.inf
    for epoch in range(args.num_epochs):
        epoch_loss = 0

        # iterate over all cells
        for i in range(num_cells):
            opt.zero_grad()
            loss = -l.forward(i) # neg log likelihood of tree embeddeding
            loss.backward() # gradient on manifold
            opt.step() # RiemannianSGD step
            epoch_loss += loss.item()
            
        # save best embedding
        if epoch_loss < best_epoch_loss:
            save_embedding(l.X, args.save_path)
            best_epoch_loss = epoch_loss
        
        # log metrics
        log_metric('epoch_loss', epoch_loss, step=epoch)
        if true_tree is not None:
            log_metric('triplets correct', cas_triplets_correct(true_tree, l.X, args.rho), step=epoch)
            log_metric('dist correlation', dist_correlation(true_tree, l.X, args.rho), step=epoch)

def save_embedding(X, save_path):
    fname = path.join(save_path, 'best_embedding.pt')
    torch.save(X, fname)
    log_artifact(fname)
    
def char_to_dist(char_matrix):
    num_cells = char_matrix.shape[0]
    num_sites = char_matrix.shape[1]
    
    dist_matrix = torch.zeros((num_cells, num_cells)) # init dist matrix
    
    # iterate over all (i,j) pairs of cells
    for i in range(num_cells):
        for j in range(i+1, num_cells):
            cur_dist = 0
            
             # sum distance btwn cells (i,j) across all sites
            for site in range(num_sites):
                x = char_matrix[i, site]
                y = char_matrix[j, site]
                cur_dist += ancestry_aware_hamming_dist(x,y)
                
            # input data into dist_matrix symmetrically
            dist_matrix[i,j] = cur_dist
            dist_matrix[j,i] = cur_dist
            
    return dist_matrix

def estimate_tree(dist_matrix, method):
    # convert numpy distance matrix to dendropy distance matrix
    file = dist_matrix.detach().numpy().astype(int)
    fname = 'dist_matrix.csv'
    np.savetxt(fname, file, delimiter = ',') # save dist matrix as csv file
    dist_matrix = phylo_dist.from_csv('dist_matrix.csv', 
                           is_first_row_column_names=False,
                           is_first_column_row_names=False
                           )
    try: # try to delete csv file
        os.remove(fname)
    except OSError as e: # raise error
        print("Error: {} - {}!".format(e.filename, e.strerror))
        
    # estimate phylogenetic tree from distance matrix    
    if method == 'upgma':
        tree = dist_matrix.upgma_tree()
    elif method == 'neighbor-joining':
        tree = dist_matrix.nj_tree()
    else:
        raise ValueError('invalid method to estimate phylogenetic tree')
    
    return tree

def embed_tree(tree, rho, num_cells, local_dim=2):
    """ embed tree into hyperbolic space with Sakar's construction

    Args:
        tree (dendropy tree): phylogenetic tree
        rho ([1x1]): negative curbature of hyperbolic space
        num_cells ([1x1]): number of cells
        local_dim (int, optional): dimension of hyperbolic surface. Defaults to 2.

    Returns:
        [num_cells x local_dim + 1]: embedding of tree in hyperbolic space
    """
    # embed tree into hyperboloid model of hyperbolic space
    # TODO: hyperboloid_wilson.py generates random samples -- add seed
    
    hyperboloid = Hyperboloid(rho.detach().numpy(), local_dim)
    tree_dict = hyperboloid.embed_tree(tree)
    
    # extract embedding of leaf cells only
    counter = 0
    X = torch.zeros(size=(num_cells, local_dim + 1))
    for key, val in tree_dict.items():
        if key.taxon is not None:
            X[counter] = torch.tensor(val)
            counter += 1            
            assert(hyperboloid.contains(val))
    return X.double()

def generate_Q(num_sites, num_states, deletion_rate, mutation_rate, indel_distribution):    
    Q = np.zeros((num_sites + num_states + 1, num_sites + num_states + 1))

    for i in range(num_sites + num_states): # fill in diagonals
        if i < num_sites:
            Q[i,i] = - (mutation_rate[i] + deletion_rate)
        else:
            Q[i,i] = - deletion_rate
            
    for i in range(num_sites): # fill in upper right
        for j in range(num_states):
            Q[i, num_sites + j] = mutation_rate[i] * indel_distribution[j]
            
    for i in range(num_sites + num_states):
        Q[i, -1] = deletion_rate
        
    return torch.tensor(Q)