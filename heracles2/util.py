import torch
import os
import numpy as np
import subprocess

from geoopt import Lorentz
from geoopt.manifolds.lorentz.math import inner
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm


from dendropy import PhylogeneticDistanceMatrix as phylo_dist
from hyperboloid_wilson import Hyperboloid

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
    
def wilson_to_geoopt(X, rho):
    """ convert hyperbolic embeddings from Wilson to geoopt convention

    The minkowski dot product <x,y> associated with hyperbolic space can be
    represented as:
        <x,y> = - (x_0 * y_0) + \Sigma_{i=1}^d x_i * y_i  (Wilson)
        <x,y> = \Sigma_{i=0}^{d-1} x_i y_i - (x_d * y_d)  (geoopt)
    The first convention is used in the Wilson paper while the second is used
    in the geoopt package.
    
    Conversion between them is necessary because I use Wilson's code to 
    isometrically embedd a tree into hyperbolic space with Sakar's construction
    and I use geoopt to carry out my hyperbolic optimization routine b/c it
    extends PyTorch's auto-differentiation capabilities.
    
    Args:
        X ([num_cells x embedding_dim]): hyperbolic embeddings in Wilson convention

    Returns:
        [num_cells x embedding_dim]: hyperbolic embeddings in geoopt convention
    """
    
    # ensure X is in Wilson hyperboloid
    check_wilson(X, torch.sqrt(rho))
    
    # convert Wilson convention to geoopt convention
    d = X.shape[1]
    idx0 = torch.tensor([d-1])
    idx1 = torch.arange(1, d-1)
    idx2 = torch.tensor([0])
    
    indices = torch.cat((idx0, idx1, idx2))
    X = X.index_select(1, indices)
    
    # ensure X is in geoopt hyperboloid
    check_geoopt(X, rho)
    
    return X

def check_wilson(X, rho):
    hyperboloid = Hyperboloid(rho.detach().numpy(), X.shape[1]-1)
    
    for i in range(X.shape[0]):
        assert(hyperboloid.contains(X[i,:].detach().numpy()))
        
def check_geoopt(X, rho):
    for i in range(X.shape[0]):
        assert(contains(X[i,:], rho, atol=1e-3))
        
def contains(v, rho, atol=1e-7):
    mdp = inner(v, v) # geoopt convention
    mdp = mdp.double()
        
    assert(v[0] > 0) # geoopt convention
    return (torch.allclose(mdp, - torch.pow(rho, 2), atol=atol) or 
            torch.allclose(mdp, - rho, atol=atol))
   
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