import torch
import numpy as np
from os import path

from geoopt.optim import RiemannianSGD
from .logalike import Logalike
from mlflow import log_metric, log_artifact

from .metrics import cas_triplets_correct, dist_correlation
from .util import generate_Q, char_to_dist, estimate_tree, embed_tree

def init(char_matrix, deletion_rate, mutation_rate, indel_distribution, 
              num_cells, num_sites, num_states, args):
    
    # reconstruct infinitesimal generator Q
    Q_list = [None] * num_sites
    for i in range(num_sites):
        Q_list[i] = generate_Q(num_sites, num_states, deletion_rate, 
                               mutation_rate, indel_distribution)
        
    # intial hyperbolic embedding
    dist_matrix = char_to_dist(char_matrix) # pairwise distancess from character matrix
    est_tree = estimate_tree(dist_matrix, method=args.tree_reconstruction) # estimate phylogenetic tree
    X = embed_tree(est_tree, torch.sqrt(args.rho), num_cells, local_dim=args.embedding_dim-1)  # embed tree into hyperbolic space

    # initalize logalike object and optimizer
    l = Logalike(X, Q_list, char_matrix, num_states, args.rho, priors=None) # TODO: add priors
    opt = RiemannianSGD([l.X], lr=args.lr, stabilize=args.stabilize)
    
    return l, opt
    
def train(char_matrix, deletion_rate, mutation_rate, indel_distribution, 
              num_cells, num_sites, num_states, args, true_tree=None):
        
    # initialize logalike object and optimizer
    l, opt = init(char_matrix, deletion_rate, mutation_rate, indel_distribution, 
              num_cells, num_sites, num_states, args)
    
    # iterate over epochs
    best_epoch_loss = np.inf
    for epoch in range(args.num_epochs):
        epoch_loss = 0

        # iterate over all cells
        for i in range(num_cells):
            opt.zero_grad()
            loss = -l.forward(i) # neg log likelihood of tree embeddeding
            loss.backward() # gradient on manifold
            opt.step() # RiemannianSGD step
            epoch_loss += loss.item()
            
        # save best embedding
        if epoch_loss < best_epoch_loss:
            save_embedding(l.X, args.save_path)
            best_epoch_loss = epoch_loss
        
        # log metrics
        log_metric('epoch_loss', epoch_loss, step=epoch)
        if true_tree is not None:
            log_metric('triplets correct', cas_triplets_correct(true_tree, l.X, args.rho), step=epoch)
            log_metric('dist correlation', dist_correlation(true_tree, l.X, args.rho), step=epoch)

def save_embedding(X, save_path):
    fname = path.join(save_path, 'best_embedding.pt')
    torch.save(X, fname)
    log_artifact(fname)