import torch
import geoopt
from geoopt import ManifoldTensor, ManifoldParameter, Lorentz
import matplotlib.pyplot as plt
import numpy as np

from .hyperboloid_wilson import Hyperboloid
from .logalike_utils import transition_matrix, wilson_to_geoopt, contains, feasible_ancestors, map_indices, stationary_dist
    
class Logalike(torch.nn.Module):
    def __init__(self, 
                 X,
                 Q_list,
                 character_matrix,
                 num_states,
                 rho,
                 priors=None):

        super().__init__()
        
        self.character_matrix = character_matrix.astype(int) # sequence data [N x N]
        
        # site-specific parameters
        self.priors = priors # TODO: not being used at the moment
        self.Q_list = Q_list # list of infinitesimal generators [Q1 ... Q_{num_sites}]
        
        self.num_cells = self.character_matrix.shape[0] # number of cells
        self.num_sites = self.character_matrix.shape[1] # number of target sites
        self.num_states = num_states # number of possible states
                
        self.rho = rho # negative curvature
        self.manifold = Lorentz(self.rho) # hyperbolic manifold
                
        # taxa location in hyperbolic space, learnable parameter [num_cells x embedding_dim]
        X = wilson_to_geoopt(X, self.rho) # Wilson to geoopt sign conversion
        self.X = ManifoldTensor(X, manifold=self.manifold) 
        self.X = ManifoldParameter(self.X)
    
    def forward(self, i):
        assert(contains(self.X[i,:], self.rho))  # ensure x_i is on hyperboloid   
        
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
                
                # map state idx [-1, 0, 1 ... M] into transition matrix P idx
                s_i, s_j, A = map_indices(s_i, s_j, A, site, self.num_sites)
                
                cur = 0
                for a in A: # iterate over all feasible ancestors
                    t1 = stationary_dist(self.num_states)
                    t2 = P[a, s_i]
                    t3 = P[a, s_j]
                    cur += t1 * t2 * t3
                    
                assert(cur > 0)
                total += torch.log(cur)
        return total