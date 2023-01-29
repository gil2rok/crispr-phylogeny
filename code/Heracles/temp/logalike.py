import torch
import geoopt

from scipy.linalg import expm

from util import transition_matrix

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
        
        # TODO: use these variable names everywhere!!
        self.num_cells = self.character_matrix.shape[0] # number of cells
        self.num_sites = self.character_matrix.shape[1] # number of target sites
        self.num_states = num_states # number of possible states
                
        self.rho = rho # negative curvature
        self.manifold = geoopt.Lorentz(self.rho) # hyperbolic manifold
        
        # taxa location in hyperbolic space, learnable parameter [N x ?]
        self.X = geoopt.ManifoldParameter(X)

    def forward(self, i):
        total = 0
        for j in range(self.num_cells): # iterate over all cells
            if j == i: continue
            
            dist = self.manifold.dist(self.X[i, :], self.X[j, :]) # geodesic btwn x_i and x_j
            for site in range(self.num_sites): # iterate over all target sites
                
                Q = self.Q_list[site] # site-specific infinitesimal generator Q 
                P = transition_matrix(dist/2, Q) # site-specific transition matrix P
                
                s_i = self.character_matrix[i, site] # state at site s for cell i
                s_j = self.character_matrix[j, site] # state at site s for cell j 
                A = feasible_ancestors(s_i, s_j, self.num_states)
                print('\t\t\tOriginal Indicies:', s_i, s_j, A)
                # map state idx [-1, 0, 1 ... M] into transition matrix P idx
                s_i, s_j, A = map_indices(s_i, s_j, A, site, self.num_sites)
                
                cur = 0
                for a in A: # iterate over all feasible ancestors
                    t1 = stationary_dist(self.num_states)
                    t2 = P[a, s_i]
                    t3 = P[a, s_j]
                    cur += t1 * t2 * t3
                
                print(cur.item(), '\t', t1, t2, t3)
                if cur.item() == 0:
                    print('\t\t\tMapped Indicies:', s_i, s_j, A, P)
                assert(torch.all(cur > 0))
                total += torch.log(cur)
        return total

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
            
        # if one state deleted and the other mutated, ancestor is unedited or mutated
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
        A (list): feasible ancestors
        num_sites (int): number of target sites
        site (int): target site at which feasible ancestors is being computed 

    Returns:
        A (list): modified feasible ancestors
    """
    
    s_i = site if s_i == 0 else s_i + num_sites
    s_j = site if s_j == 0 else s_j + num_sites
    A = [site if a == 0 else a + num_sites for a in A]
    
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
    return pi_si