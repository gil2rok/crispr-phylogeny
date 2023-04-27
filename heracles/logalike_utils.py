import torch
import os
import numpy as np
import subprocess

from geoopt import Lorentz
from geoopt.manifolds.lorentz.math import inner
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm

from .hyperboloid_wilson import Hyperboloid

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
    
def feasible_ancestors(s_i, s_j, num_states):
    """ generate feasible ancestors for states s_i, s_j
    TODO: may need to be vectorized
    
    Args:
        s_j [1] : state at site s for cell j
        s_i [1] : state at site s for cell i
        num_states (int): number of possible mutation states

    Returns:
        A ([1 x ?]): feasible ancestors of states s_i, s_j
    """
    
    # TODO: transform ancester set A from a tensor to a list -- no need for backprop info
        
    # if states s_i and s_j are both unedited, their ancestor is unedited
    if s_i == s_j == 0:
        A = torch.tensor([0])
    
    # if have a deleted state
    elif s_i == -1 or s_j == -1:
        
        # if both states are deleted, ancestor may be unedited or mutations 1...M
        # ancestor may not be deleted -- continue to previous ancestor
        if s_i == -1 and s_j == -1:
            A = torch.arange(0, num_states +1)
            
        # if one state deleted and the other unedited, ancestor is unedited
        elif s_i == 0 or s_j == 0:
            A = torch.tensor([0])
            
        # if one state deleted and the other mutated, ancestor is unedited (0) or mutated
        else:
            A = torch.tensor([0, s_i + s_j])
       
    # if don't have a deleted state 
    else:
        # if states are the same mutation, ancestor is unedited or has same mutation
        if s_i == s_j:
            A = torch.tensor([0, s_i])
        
        # if states are different mutations, ancestor must be unedited
        else:
            A = torch.tensor([0])
        
    return A

def map_indices(s_i, s_j, A, site, num_sites):
    """ map feasible ancestors to correct indicies in transition matrix P

    Args:
        A (tensor): feasible ancestors
        num_sites (int): number of target sites
        site (int): target site at which feasible ancestors is being computed 

    Returns:
        A (list): modified feasible ancestors
    """
    
    if s_i == 0:
        s_i = site
    elif s_i == -1:
        s_i = -1
    else:
        s_i += num_sites - 1
        
    if s_j == 0:
        s_j = site
    elif s_j == -1:
        s_j = -1
    else:
        s_j += num_sites - 1
        
    A = [site if a == 0 else a + num_sites - 1 for a in A] #TODO: check if correct
    
    return s_i, s_j, A

def stationary_dist(num_states):
    """ probability of state s_i according to stationary distribution
    TODO: fix assumption that stationary distribution is uniform, instead use Felsenstein’s algorithm
            Sitatra already implmented this in est_lca_priors
    
    Specifically return the probability of observing 
    state s_i at site s for cell i according to the 
    stationary distribution pi, defined by the Continous Time
    Markov Chain of CRISPR evolution.
    
    Assume pi is uniform distribution over all M possible
    mutation states: {0, 1 ... M_s, D}.

    Args:
        S (int): num possible states (S = |\sigma_i| = |{0, 1 ... M_{\sigma_i}, D}| )
 
    Returns:
        [1]: probability of base s_i under stationary distribution pi
    """
    
    pi_si = 1 / num_states
    return torch.tensor([pi_si])