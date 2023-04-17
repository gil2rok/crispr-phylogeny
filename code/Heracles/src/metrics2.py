import numpy as np
from geoopt import Lorentz
from collections import Counter
from itertools import combinations
import torch
from scipy.stats import pearsonr
from scipy.special import comb

def embeddings_to_dist(X, rho):
    """ estimated pairwise distances between cell embeddings (along hyperboloic geodescics)

    Args:
        X (np [num_cells x num_sites]): hyperbolic embedding of cells
        rho (float): negative curvate of Lorentz manifold

    Returns:
        np [num_cells x num_cells]: estimated pairwise distances
    """
    
    num_cells = X.shape[0] # TODO: ensure X is repeated appropriately
    manifold = Lorentz(rho)
    est_dist = np.zeros((num_cells, num_cells)) # [num_cells x num_cells]
    
    # iterate over all pairs of cells
    for i in range(num_cells):
        for j in range(i+1, num_cells):
            # TODO: ensure manifold dist is a numpy array, not torch tensor!
            est_dist[i,j] = manifold.dist(X[i,:], X[j,:]) # geodesic btwn x_i and x_j
            est_dist[j, i] = est_dist[i,j] # symmetric matrix entry            
    return est_dist

def tree_to_dist(true_tree):
    """ true pairwise distances between cells along phylogenetic tree

    Args:
        true_tree (Cassieopia tree): true phylogenetic tree

    Returns:
        np [num_cells x num_cells]: true pairwise distances
    """
    
    num_cells = true_tree.character_matrix.shape[0]
    true_dist = np.zeros((num_cells, num_cells)) # [num_cells x num_cells]
    
    # iterate over all pairs of cells
    for i, leaf in enumerate(true_tree.leaves):
        
        # tree distances from leaf to all other leaves
        true_dist[i, :] = list(
            true_tree.get_distances(leaf, leaves_only=True).values()
        )
    
    return true_dist

def repeat_embeddings(X, cm):
    """ repeat embeddings for cells with same state in a character matrix

    Args:
        X (np [num_cells x embedding_dim]): hyperbolic embeddings of cells
        char_matrix (np [num_cells x num_sites]): character matrix of cells

    Returns:
        np [num_cells_repeat x embedding_dim]: hyperbolic embeddings of cells repeated
    """
    
    counts_dict = Counter([tuple(row) for row in cm]) # dict[cell cassette] --> num times cell's casette appears in cm
    counts = torch.tensor(np.fromiter(counts_dict.values(), dtype=int)) # convert dict to torch tensor
    return torch.repeat_interleave(X, counts, axis=0)
    
def triplets_correct(true_tree, X, rho):
    """ compute triplets correct btwn true and estimated trees
    
    Iterate over all 3-node subtrees comprised of leaf cells. Count how many
    3-node subtrees have topologies that differ btwn true and estimated trees. 
    
    There are four possible topologies for subtree with leaf-nodes a,b,c:
        (1) a,b share a parent
        (2) a,c share a parent
        (3) b,c share a parent
        (4) a,b,c share a parent
    Read https://academic.oup.com/sysbio/article/45/3/323/1616252 for more 
    info.
    
    Args:
        true_tree (Cassieopia tree): true phylogenetic tree
        X (np [num_cells x embedding_dim]): hyperbolic cell embeddings
        rho (float): negative curvate of Lorentz manifold

    Returns:
        int: number of triplets correct in X
    """
    
    # preprocess
    X = repeat_embeddings(X, true_tree.character_matrix.to_numpy())    
    est_dist = embeddings_to_dist(X, rho)
    true_dist = tree_to_dist(true_tree)
    
    num_cells = X.shape[0]
    correct = 0 # num triplets correct
    
    # iterate over all triplets of cells
    for i, j, k in combinations(range(num_cells), 3):
        td1, td2, td3 = true_dist[i,j], true_dist[i,k], true_dist[j,k] # true distances
        ed1, ed2, ed3 = est_dist[i,j], est_dist[i,k], est_dist[j,k] # estimated distances
        
        # compare toplogy of all three-node subtrees in true and estimated trees
        # use true and estimated distance to check all four possible topologies
        topology1 = (td1 < td2) and (ed1 < ed2)
        topology2 = (td1 < td3) and (ed1 < ed3)
        topology3 = (td2 < td3) and (ed2 < ed3)
        topology4 = (td1 == td2 == td3) and (ed1 == ed2 == ed3)
        
        if  topology1 or topology2 or topology3 or topology4:
            correct += 1
    return correct / comb(num_cells, 3) # normalize by number of triplets

def dist_correlation(true_tree, X, rho):
    """ compute correlation btwn true and estimated pairwise distances

    Args:
        true_tree (Cassiopiea tree): true phylogenetic tree
        X (tensor [num_cells x num_sites]): hyperbolic cell embeddings
        rho (float): negative curvate of Lorentz manifold

    Returns:
        float: correlation btwn true and estimated pairwise distances
    """
    
    # preprocess
    X = repeat_embeddings(X, true_tree.character_matrix.to_numpy())    
    est_dist = embeddings_to_dist(X, rho)
    true_dist = tree_to_dist(true_tree)
    
    # correlation btwn true and estimated distances
    return pearsonr(est_dist.flatten(), true_dist.flatten()).statistic
    
                    
                
   