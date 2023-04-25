import numpy as np
import torch
import copy

from geoopt import Lorentz
from collections import Counter, defaultdict
from itertools import combinations
from scipy.stats import pearsonr
from scipy.special import comb
from typing import Dict, Tuple
from cassiopeia.critique import critique_utilities
from cassiopeia.data import CassiopeiaTree


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
    TODO: confirm assumption that ordering in character matrix is same as ordering in X
            meaning that X[i,:] corresponds to cm[i,:]
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
    Read https://academic.oup.com/sysbio/article/45/3/323/1616252 and
    https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-14-S2-S18 
    for more 
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
    
def cas_triplets_correct(true_tree, X, rho, all_triplets=True):
    # preprocess
    X = repeat_embeddings(X, true_tree.character_matrix.to_numpy())    
    est_dist = embeddings_to_dist(X, rho)
    triplets = _cas_triplets_correct(true_tree, est_dist)
    
    if all_triplets: # all triplets correct
        return np.mean(list(triplets[0].values()))
    else: # resolved triplets correct
        return np.mean(list(triplets[1].values()))

def get_outgroup(est_dist, triplet, cm):
    i, j, k = triplet
    a, b, c = [cm.index.get_loc(el) for el in triplet] # map cell names (i,j,k) to est_dist indices (a,b,c)
    ed1, ed2, ed3 = est_dist[a,b], est_dist[b,c], est_dist[a,c] # estimated distances
    
    if ed2 <= ed1 and ed2 <= ed3:
        return i
    elif ed3 <= ed1 and ed3 <= ed2:
        return j
    elif ed1 <= ed2 and ed1 <= ed3: 
        return k
                    
def _cas_triplets_correct(
    tree1: CassiopeiaTree,
    est_dist: np.ndarray,
    number_of_trials: int = 1000,
    min_triplets_at_depth: int = 1,
) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]
]:
    """ Modified Cassiopeia function to calculate the triplets correct accuracy between two trees.

    Takes in two newick strings and computes the proportion of triplets in the
    tree (defined as a set of three leaves) that are the same across the two
    trees. This procedure samples the same number of triplets at every depth
    such as to reduce the amount of bias of sampling triplets randomly.

    Args:
        tree1: Input CassiopeiaTree
        est_dist [num_cells x num_cells]: estimated pairwise distances btwn hyperbolic cell embeddings
        number_of_trials: Number of triplets to sample at each depth
        min_triplets_at_depth: The minimum number of triplets needed with LCA
            at a depth for that depth to be included

    Returns:
        Four dictionaries storing triplet information at each depth:
            all_triplets_correct: the total triplets correct
            resolvable_triplets_correct: the triplets correct for resolvable
                triplets
            unresolved_triplets_correct: the triplets correct for unresolvable
                triplets
            proportion_resolvable: the proportion of unresolvable triplets per
                depth
    """

    # keep dictionary of triplets correct
    all_triplets_correct = defaultdict(int)
    unresolved_triplets_correct = defaultdict(int)
    resolvable_triplets_correct = defaultdict(int)
    proportion_unresolvable = defaultdict(int)

    # create copies of the trees and collapse process
    T1 = copy.copy(tree1)
    T1.collapse_unifurcations()

    # set depths in T1 and compute number of triplets that are rooted at
    # ancestors at each depth
    depth_to_nodes = critique_utilities.annotate_tree_depths(T1)

    max_depth = np.max([T1.get_attribute(n, "depth") for n in T1.nodes])
    for depth in range(max_depth):

        score = 0
        number_unresolvable_triplets = 0

        # check that there are enough triplets at this depth
        candidate_nodes = depth_to_nodes[depth]
        total_triplets = sum(
            [T1.get_attribute(v, "number_of_triplets") for v in candidate_nodes]
        )
        if total_triplets < min_triplets_at_depth:
            continue

        for _ in range(number_of_trials):

            (i, j, k), out_group = critique_utilities.sample_triplet_at_depth(
                T1, depth, depth_to_nodes
            )
            
            reconstructed_outgroup = get_outgroup(
                est_dist, (i, j, k), T1.character_matrix
            )

            is_resolvable = True
            if out_group == "None":
                number_unresolvable_triplets += 1
                is_resolvable = False

            # increment score if the reconstructed outgroup is the same as the
            # ground truth
            score = int(reconstructed_outgroup == out_group)

            all_triplets_correct[depth] += score
            if is_resolvable:
                resolvable_triplets_correct[depth] += score
            else:
                unresolved_triplets_correct[depth] += score

        all_triplets_correct[depth] /= number_of_trials

        if number_unresolvable_triplets == 0:
            unresolved_triplets_correct[depth] = 1.0
        else:
            unresolved_triplets_correct[depth] /= number_unresolvable_triplets

        proportion_unresolvable[depth] = (
            number_unresolvable_triplets / number_of_trials
        )

        if proportion_unresolvable[depth] < 1:
            resolvable_triplets_correct[depth] /= (
                number_of_trials - number_unresolvable_triplets
            )
        else:
            resolvable_triplets_correct[depth] = 1.0

    return (
        all_triplets_correct,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    )
