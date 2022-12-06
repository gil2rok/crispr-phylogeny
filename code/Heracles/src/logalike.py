import torch
import geoopt

from scipy.linalg import expm

class Logalike(torch.nn.Module):
    def __init__(self, 
                 rho,
                 character_matrix,
                 init_points,
                 num_mutations,
                 S,):

        super().__init__()
        
        self.X_init = init_points # init location of cells [N x ?]
        self.character_matrix = character_matrix # sequence data [N x N]
        
        # TODO: use these variable names everywhere!!
        self.num_cells = self.character_matrix.shape[0] # number of cells
        self.num_sites = self.character_matrix.shape[1] # number of target sites
        self.num_mutations = num_mutations # num feasible mutations at each site
        self.num_states = S # number of possible states
                
        self.rho = rho # negative curvature
        self.manifold = geoopt.Lorentz(self.rho) # hyperbolic manifold
        
        # taxa location in hyperbolic space, learnable parameter [N x ?]
        self.X = geoopt.ManifoldParameter(
            self.X_init
        )

    def forward(self, Q, i):
        total = 0
        for j in range(self.num_cells): # iterate over all cells
            if j == i: continue
            
            dist = self.manifold.dist(self.X[i:i+1], self.X[j:j+1]) # geodesic btwn x_i and x_j
            P = transition_matrix(dist/2, Q) # compute transition matrix
            print(torch.count_nonzero(P))
            
            for site in range(self.num_sites): # iterate over all target sites
                s_i = self.character_matrix[i, site] # state at site s for cell i
                s_j = self.character_matrix[j, site] # state at site s for cell j
                
                cur = 0
                A = feasible_ancestors(s_i, s_j, self.num_mutations[site])
                for a in A: # iterate over all feasible ancestors
                    t1 = stationary_dist(self.num_states)
                    t2 = P[a, s_i]
                    t3 = P[a, s_j]
                    cur += t1 * t2 * t3
                
                print(cur.item(), '\t', t1, t2.item(), t3.item())
                assert(torch.all(cur > 0))
                total += torch.log(cur)
        return total / self.num_cells

def transition_matrix(t, Q):
    """ compute transition matrix P
    TODO: may need to be vectorized
    TODO: convert to compact computation
    
    s_i represents the state at site sigma for
    cell i. Furthermore, P_{s_i, s_j}(t) represents 
    the conditional probability of observing
    state s_j t "time units" after state s_i.
    
    Args:
        t [1x1]: time
        Q [NxN]: infinitesimal generator

    Returns:
        P [NxN]: transition matrix
    """
    
    P = torch.matrix_exp(Q * t)
    return P

def feasible_ancestors(s_i, s_j, num_mutations):
    """ generate feasible ancestors for states s_i, s_j
    TODO: may need to be vectorized
    
    Args:
        s_j [1] : state at site s for cell j
        s_i [1] : state at site s for cell i

    Returns:
        A ([1 x ?]): feasible ancestors of states s_i, s_j
    """
    
    # if states s_i and s_j are both unedited, their ancestor is unedited
    if s_i == s_j == 0:
        A = torch.tensor([0])
    
    # if have a deleted state
    elif s_i == 'D' or s_j == 'D':
        
        # if both states are deleted, ancestor may be unedited, mutations 1...M, or deleted
        if s_i == 'D' and s_j == 'D':
            mutations = torch.arange(1, num_mutations+1)
            A = torch.cat(torch.tensor([0]),
                          mutations,
                          torch.tensor(['D']))
            
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

def stationary_dist(S):
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
    pi_si = 1 / S
    return pi_si
