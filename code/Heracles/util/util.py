import numpy as np

def extract_compact_Q(mutation_rate, deletion_rate):
    """ extract compact representation of infinitesimal generator Q
    
    Better def of function arguments described in sitara-writeup.pdf 
    section 1.3.1 Stucture of the Transition Matrix

    Args:
        mutation_rate [num_sites x 1]: per-site rate of transition to some mutated state
        deletion_rate [1 x 1]: rate of transition to a deleted state
    """
    num_sites = len(mutation_rate)
    
    t1 = np.vstack(( np.diag(-(mutation_rate) + deletion_rate), np.zeros((2, num_sites)) )) 
    t2 = np.concatenate((mutation_rate, deletion_rate, np.array([0])))
    t3 = np.concatenate(( deletion_rate.repeat(num_sites + 1), np.array([0])))
    
    Q_compact = np.hstack((t1, t2[:, np.newaxis], t3[:, np.newaxis]), )
    return Q_compact