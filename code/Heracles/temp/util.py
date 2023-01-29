import torch
from scipy.linalg import expm


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
    
    P =  expm(t.detach().numpy() * Q)
    return torch.tensor(P, dtype=torch.float64)

    # P = expm(t * Q) # Scipy's matrix exponentiation function
    # return torch.tensor(P) # convert to tensor for auto-differentiation