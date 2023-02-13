import torch
import numpy as np

from dendropy import PhylogeneticDistanceMatrix as phylo_dist
from util.hyperboloid_wilson import Hyperboloid

def transition_matrix(t, Q):
    """ compute transition matrix P from infinitesimal generator Q and time t

    Let s_i represent the state at site sigma for cell i. Then
    P_{s_i, s_j}(t) represents the conditional probability of observing
    state s_j t "time units" after state s_i.
    
    Args:
        t [1x1]: time, approximated by hyperbolic distance
        Q [num_states+num_sites+1 x num_states+num_sites+1]: infinitesimal generator Q

    Returns:
        P [num_states+num_sites+1 x num_states+num_sites+1]: transition matrix P
    """
    
    P = torch.matrix_exp(t * Q)
    return P

def ancestry_aware_hamming_dist(x, y):
    # dist(0,0) = 0, dist(x,x) = 1, dist(x,y) = 2
    
    if x == y: # x and y are the same
        return torch.tensor([0])
    
    elif x == 0 or y == 0: # x and y are different and one is unedited
        return torch.tensor([1])
    
    elif x != y: # x and y are different and both are edited
        return torch.tensor([2])
    
    else:
        raise ValueError("Ancestry aware hamming distance error")
    
def char_matrix_to_dist_matrix(char_matrix):
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
    np.savetxt('dist_matrix.csv', file, delimiter = ',') # save dist matrix
    dist_matrix = phylo_dist.from_csv('dist_matrix.csv', 
                           is_first_row_column_names=False,
                           is_first_column_row_names=False
                           )
    
    # estimate phylogenetic tree from distance matrix
    if method == 'upgma':
        tree = dist_matrix.upgma_tree()
    elif method == 'neighbor-joining':
        tree = dist_matrix.nj_tree()
    else:
        raise ValueError('invalid method for estimate phylogenetic tree')
    
    return tree

def embed_tree(tree, rho, num_cells, local_dim=2):
    hyperboloid = Hyperboloid(rho.detach().numpy(), local_dim)
    tree_dict = hyperboloid.embed_tree(tree)
    
    counter = 0
    X = torch.zeros(size=(num_cells, local_dim + 1))
    for key, val in tree_dict.items():
        if key.taxon is not None:
            X[counter] = torch.tensor(val)
            counter += 1
    return X

    