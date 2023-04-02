import torch
import geoopt
import matplotlib.pyplot as plt
import numpy as np

from util.hyperboloid_wilson import Hyperboloid
from util.misc_util import transition_matrix

def swap(X):
    d = X.shape[1]
    idx0 = torch.tensor([d-1])
    idx1 = torch.arange(1, d-1)
    idx2 = torch.tensor([0])
    
    indices = torch.cat((idx0, idx1, idx2))
    X = X.index_select(1, indices)
    return X

def check_wilson(X, rho):
    hyperboloid = Hyperboloid(rho.detach().numpy(), X.shape[1]-1)
    
    for i in range(X.shape[0]):
        assert(hyperboloid.contains(X[i,:].detach().numpy()))
        
def check_geoopt(X, rho):
    for i in range(X.shape[0]):
        assert(contains(X[i,:], rho))
    
class Logalike(torch.nn.Module):
    def __init__(self, 
                 X,
                 priors,
                 Q_list,
                 character_matrix,
                 num_states,
                 rho):

        super().__init__()
        
        self.character_matrix = character_matrix # sequence data [N x N]
        
        # site-specific parameters
        self.priors = priors # TODO: not being used at the moment
        self.Q_list = Q_list # list of infinitesimal generators [Q1 ... Q_{num_sites}]
        
        self.num_cells = self.character_matrix.shape[0] # number of cells
        self.num_sites = self.character_matrix.shape[1] # number of target sites
        self.num_states = num_states # number of possible states
                
        self.rho = rho # negative curvature
        self.manifold = geoopt.Lorentz(self.rho) # hyperbolic manifold
        
        # taxa location in hyperbolic space, learnable parameter [num_cells x embedding_dim]
        check_wilson(X, self.rho) # ensure X is in Wilson hyperboloid
        X = geoopt.ManifoldTensor(swap(X), manifold=self.manifold)
        self.X = geoopt.ManifoldParameter(X)
        check_geoopt(self.X, self.rho) # ensure X is NOW in geoopt hyperboloid
        
    def forward(self, i):
        total = 0
        for j in range(self.num_cells): # iterate over all cells
            if j == i: continue
            
            dist = self.manifold.dist(self.X[i, :], self.X[j, :]) # geodesic btwn x_i and x_j
            assert(contains(self.X[i,:], self.rho ))     

            for site in range(self.num_sites): # iterate over all target sites
                
                Q = self.Q_list[site] # site-specific infinitesimal generator Q 
                P = transition_matrix(dist/2, Q) # site-specific transition matrix P
                
                s_i = self.character_matrix[i, site] # state at site s for cell i
                s_j = self.character_matrix[j, site] # state at site s for cell j 
                A = feasible_ancestors(s_i, s_j, self.num_states)
                
                # map state idx [-1, 0, 1 ... M] into transition matrix P idx
                s_i, s_j, A = map_indices(s_i, s_j, A, site, self.num_sites)
                
                cur = 0
                for a in A: # iterate over all feasible ancestors
                    t1 = stationary_dist(self.num_states)
                    t2 = P[a, s_i]
                    t3 = P[a, s_j]
                    cur += t1 * t2 * t3
                    
                total += torch.log(cur)
        return total
    
    def manual_grad(self, i):
        total = 0
        for j in range(self.num_cells): # iterate over num cells
            dist = self.manifold.dist(self.X[i, :], self.X[j, :]) # geodesic btwn x_i and x_j
            for site in range(self.num_sites): # iterate over all target sites
                
                Q = self.Q_list[site] # site-specific infinitesimal generator Q 
                P = transition_matrix(dist/2, Q) # site-specific transition matrix P
                
                s_i = self.character_matrix[i, site] # state at site s for cell i
                s_j = self.character_matrix[j, site] # state at site s for cell j 
                A = feasible_ancestors(s_i, s_j, self.num_states)
                
                # map state idx [-1, 0, 1 ... M] into transition matrix P idx
                s_i, s_j, A = map_indices(s_i, s_j, A, site, self.num_sites)
                
                num, den = 0, 0
                for a in A: # iterate over all feasible ancestors
                    t1 = stationary_dist(self.num_states)
                    t2 = P[a, s_i]
                    t3 = P[a, s_j]
                    
                    num += t1 * t2 * t3 * (Q[a, s_i] + Q[a, s_j])
                    den += t1 * t2 * t3
                    
                den *= 2
                dist_grad = self.grad_dist(self.X[i,:], self.X[j,:])
                total += (num / den) * dist_grad
        return total
                
    def grad_dist(self, u, v):
        num = torch.pow(self.rho, -2) * _inner(u, v) * u - v
        temp = (torch.pow(self.rho, -1) * _inner(u, v)) ** 2
        den = torch.sqrt(temp - torch.pow(self.rho, 2))
        return num / den

def minkowski_dot(x, y):
    # return -(x[0:1] @ y[0:1]) + (x[1:] @ y[1:]) # geoopt convention
    return (x[:-1] @ y[:-1]) - (x[-1:] @ y[-1:]) # Wilson convention

def my_dist(x, y, k):
    # wilson implementation of hyperbolic distance
    
    mkd = minkowski_dot(x,y)
    arg = -mkd / k.pow(2)
    arg = torch.clamp(arg, min=torch.tensor([1])) # proper clamping
    return k * torch.acosh(arg) # valid domain is [1, inf]

def contains(v, rho, atol=1e-7):
    mdp = _inner(v, v) # geoopt convention
    mdp = mdp.double()
    
    mdp1 = mdp.detach().numpy()
    rho1 = -torch.pow(rho, 2).detach().numpy() 
    
    # ic(mdp1)
    # ic(rho1)
    
    assert(v[0] > 0) # geoopt convention
    return torch.allclose(mdp, - torch.pow(rho, 2), atol=atol) or torch.allclose(mdp, - rho, atol=atol)

###### start of not my code #####
def _dist(x, y, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    d = -_inner(x, y, dim=dim, keepdim=keepdim)
    return torch.sqrt(k) * arcosh(d / k)

def _inner(u, v, keepdim: bool = False, dim: int = -1):
    d = u.size(dim) - 1
    uv = u * v
    if keepdim is False:
        return -uv.narrow(dim, 0, 1).sum(dim=dim, keepdim=False) + uv.narrow(
            dim, 1, d
        ).sum(dim=dim, keepdim=False)
    else:
        return torch.cat((-uv.narrow(dim, 0, 1), uv.narrow(dim, 1, d)), dim=dim).sum(
            dim=dim, keepdim=True
        )
        
def arcosh(x: torch.Tensor):
    dtype = x.dtype
    x = torch.max(x, torch.tensor([1])) # TODO: evaluate if change is necessary
    z = torch.sqrt(torch.clamp_min(x.double().pow(2) - 1.0, 1e-15)) # clamp_min is equivalent to max
    temp = torch.log(x + z).to(dtype)
    return temp

###### end of not my code ######

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
        
          
    # s_i = site if s_i == 0 else s_i + num_sites
    # s_j = site if s_j == 0 else s_j + num_sites
    # A = [site if a == 0 else a + num_sites for a in A]
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