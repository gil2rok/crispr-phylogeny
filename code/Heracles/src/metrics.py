import torch
import numpy as np

from collections import Counter
from scipy.stats import pearsonr
from geoopt import Lorentz
from scipy.linalg import norm
from scipy.spatial.distance import pdist, squareform

from utils import estimate_tree

def manifold_dist(u, v, rho):
    u = torch.tensor(u)
    v = torch.tensor(v)
    
    manifold = Lorentz(k=rho)
    dist = manifold.dist(u, v)
    return dist.detach().numpy()

def num_unique(true_tree):
    char_matrix = true_tree.character_matrix.to_numpy() # character matrix of leaf cells
    num_sites = char_matrix.shape[1]
    keys = tuple([char_matrix[:,i] for i in range(num_sites -1, -1, -1)])
    char_matrix = char_matrix[np.lexsort(keys)]
    __, unique_counts = np.unique(char_matrix, return_counts=True, axis=0)
    return unique_counts

def pairwise_dist(true_tree, X, rho):
    # preprocess
    X = X.detach().numpy()
    char_matrix = true_tree.character_matrix.to_numpy()
    counts = Counter([tuple(row) for row in char_matrix]) # counts[cell state] --> count of cell state

    # add duplicates to X according to original data
    X = np.repeat(X, counts.values(), axis=0)
    est_dist = squareform(pdist(X, metric=manifold_dist, rho=rho))
    
    num_cells = len(true_tree.leaves)
    true_dist = np.empty((num_cells, num_cells))
    for i, leaf in enumerate(true_tree.leaves):
        true_dist[i] = list(true_tree.get_distances(leaf, leaves_only=True).values())
        
    # diff = np.abs(true_dist - est_dist)
    # error = norm(diff, ord='fro')
    error = pearsonr(true_dist.reshape(-1), est_dist.reshape(-1))
    return error.statistic

def robinson_foulds(true_tree, X, rho, method):
    pass