import torch
import numpy as np
import cassiopeia as cas

from collections import Counter
from scipy.stats import pearsonr
from numba import jit
from geoopt import Lorentz
from scipy.linalg import norm
from scipy.spatial.distance import pdist, squareform
from cassiopeia.critique import robinson_foulds, triplets_correct

from util.util import estimate_tree

def manifold_dist(u, v, rho):
    u = torch.tensor(u)
    v = torch.tensor(v)
    
    manifold = Lorentz(k=rho)
    dist = manifold.dist(u, v)
    return dist.detach().numpy()

def est_pairwise_dist(char_matrix, X, rho):
    # preprocess
    X = X.detach().numpy()
    counts_dict = Counter([tuple(row) for row in char_matrix]) # counts[cell state] --> count of cell state
    counts = np.fromiter(counts_dict.values(), dtype=int) # convert dict to np array

    # add duplicates to X according to original data
    X = np.repeat(X, counts, axis=0)
    est_dist = squareform(pdist(X, metric=manifold_dist, rho=rho))
    return est_dist

def dissimilarity_function_general(rho, embedding_dict, s1, s2, missing_state_indicator=-1, nb_weights=None,):
    x1 = embedding_dict[tuple(s1)]
    x2 = embedding_dict[tuple(s2)]
    
    return manifold_dist(x1, x2, rho)
    
def dist_correlation(true_tree, X, rho):
    char_matrix = true_tree.character_matrix.to_numpy()
    init_cell = np.zeros((1, char_matrix.shape[1]))
    char_matrix = np.concatenate((char_matrix, init_cell), axis=0) # add unmodified cell
    est_dist = est_pairwise_dist(char_matrix, X, rho)
    
    num_cells = len(true_tree.leaves) + 1
    true_dist = np.empty((num_cells, num_cells))
    for i, leaf in enumerate(true_tree.leaves):
        true_dist[i, :-1] = list(true_tree.get_distances(leaf, leaves_only=True).values())
   
    true_dist[-1, :-1] = list(true_tree.get_distances(true_tree.root, leaves_only=True).values())
    true_dist[:-1, -1] = true_dist[-1, :-1]
    
    error = pearsonr(true_dist.reshape(-1), est_dist.reshape(-1)).statistic
    return error

def rf(true_tree, X, rho, priors=None):
    # init est tree
    char_matrix = true_tree.character_matrix.copy(deep=True)
    est_tree = cas.data.CassiopeiaTree(character_matrix=char_matrix, priors=priors)
    
    # solve est tree with neighbor joining solver
    embedding_dict = dict()
    char_matrix = char_matrix.drop_duplicates().to_numpy()
    char_matrix = np.concatenate((char_matrix, np.zeros((1, char_matrix.shape[1]))), axis=0) # add unmodified cell
    char_matrix = [tuple(cm) for cm in char_matrix]
    for cell, embedding in zip(char_matrix, X):
        embedding_dict[cell] = embedding
        
    @jit(nopython=False)
    def dissimilarity_function(s1, s2, missing_state_indicator, nb_weights):
        dissimilarity_function_general(rho, embedding_dict, s1, s2, missing_state_indicator, nb_weights)
    
    nj_solver = cas.solver.NeighborJoiningSolver(dissimilarity_function=dissimilarity_function, add_root=True)
    nj_solver.solve(est_tree, collapse_mutationless_edges=True)
    
    rf, rf_max = robinson_foulds(true_tree, est_tree)
    return rf / rf_max # normalize score

def tc2(true_tree, X, rho):
    char_matrix = true_tree.character_matrix.to_numpy()
    num_cells = char_matrix.shape[0]
    num_sites = char_matrix.shape[1]
    
    init_cell = np.zeros((1, num_sites))
    char_matrix = np.concatenate((char_matrix, init_cell), axis=0) # add unmodified cell
    est_dist = est_pairwise_dist(char_matrix, X, rho)
    
    num_cells = len(true_tree.leaves) + 1
    true_dist = np.empty((num_cells, num_cells))
    for i, leaf in enumerate(true_tree.leaves):
        true_dist[i, :-1] = list(true_tree.get_distances(leaf, leaves_only=True).values())
   
    true_dist[-1, :-1] = list(true_tree.get_distances(true_tree.root, leaves_only=True).values())
    true_dist[:-1, -1] = true_dist[-1, :-1]

    tc = 0
    for i in range(num_cells):
        for j in range(i+1, num_cells):
            for k in range(j+1, num_cells):
                td1, td2, td3 = true_dist[i, j], true_dist[j, k], true_dist[k, i]
                ed1, ed2, ed3 = est_dist[i, j], est_dist[j, k], est_dist[k, i]
                
                # check all four topologies for how three-node subtrees can be equal
                topology1 = (td1 < td2) == (ed1 < ed2)
                topology2 = (td2 < td3) == (ed2 < ed3)
                topology3 = (td3 < td1) == (ed3 < ed1)
                topology4 = (td1 == td2 == td3) == (ed1 == ed2 == ed3)
                
                if not (topology1 or topology2 or topology3 or topology4):
                    tc += 1
    return tc
              
def tc(true_tree, X, rho, priors=None):
    
    # init est tree
    char_matrix = true_tree.character_matrix.copy(deep=True)
    est_tree = cas.data.CassiopeiaTree(character_matrix=char_matrix, priors=priors)
    
    # solve est tree with neighbor joining solver
    embedding_dict = dict()
    char_matrix = char_matrix.drop_duplicates().to_numpy()
    char_matrix = np.concatenate((char_matrix, np.zeros((1, char_matrix.shape[1]))), axis=0) # add unmodified cell
    char_matrix = [tuple(cm) for cm in char_matrix]
    for cell, embedding in zip(char_matrix, X):
        embedding_dict[cell] = embedding
        
    @jit(nopython=False)
    def dissimilarity_function(s1, s2, missing_state_indicator, nb_weights):
        dissimilarity_function_general(rho, embedding_dict, s1, s2, missing_state_indicator, nb_weights)
    
    nj_solver = cas.solver.NeighborJoiningSolver(dissimilarity_function=dissimilarity_function, add_root=True)
    nj_solver.solve(est_tree, collapse_mutationless_edges=True)
    
    all_tc, _, _, _ = triplets_correct(true_tree, est_tree)
    return np.mean(list(all_tc.values()))