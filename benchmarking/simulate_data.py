import os
import pickle
import numpy as np
import cassiopeia as cas

from cassiopeia.sim import BirthDeathFitnessSimulator
from cassiopeia.sim import Cas9LineageTracingDataSimulator

seed = 0
rng = np.random.default_rng(seed)
path = 'data' # relative path to data directory

def simulate_evolutionary_tree():
    # instantiate BirthDeathFitnessSimulator() object and simulate tree
    bd_sim = cas.sim.BirthDeathFitnessSimulator(
        birth_waiting_distribution = lambda scale: rng.exponential(scale),
        initial_birth_scale = 0.5,
        death_waiting_distribution = lambda: rng.exponential(1.5),
        mutation_distribution = lambda: 1 if rng.uniform() < 0.5 else 0,
        fitness_distribution = lambda: rng.normal(0, .5),
        fitness_base = 1.3,
        num_extant = 20,
        random_seed=seed
    )
    true_tree = bd_sim.simulate_tree()
    
    # uncomment below to plot phylogenetic tree
    # fig = cas.pl.plot_plotly(true_tree, random_state=seed)
    # fig.show()
    # TODO: use fig.write_html('path')
    
    return true_tree

def lineage_tracing(true_tree, params, num_sites, num_states):
   # instantiate Cas9 lineage tracing object & overlay data onto ground_truth_tree
    lt_sim = cas.sim.Cas9LineageTracingDataSimulator(
        number_of_cassettes = 1,
        size_of_cassette = num_sites,
        mutation_rate = params['mutation_rate'],
        state_generating_distribution = None,
        number_of_states = num_states,
        state_priors = params['transition_prob'], # must be dict
        heritable_silencing_rate = params['deletion_rate'],
        stochastic_silencing_rate = 0.1,
        heritable_missing_data_state = -1,
        stochastic_missing_data_state = -1,
        random_seed = seed
    )
    lt_sim.overlay_data(true_tree)    

def simulate_data(transition_prob, num_sites=5, num_states=15, mutation_rate=0.4, deletion_rate=9e-4, path=path):    
    # hyper-parameters and parameters
    
    params = {'mutation_rate': np.repeat(mutation_rate, num_sites),     # mutation rates [λM_1, λM_2, ..., λM_NumSites]
              'deletion_rate': np.array([deletion_rate]),               # deletion rate λD
              'transition_prob': {i: transition_prob[i] for i in range(num_states)}}               # simplex P = [p_1 ... p_NumStates]
              # ^ probability p_i of transitioning from unedited state to mutated state i

    # simulate evolutionary tree and overly CRISPR-Cas9 data on top of it
    true_tree = simulate_evolutionary_tree()
    lineage_tracing(true_tree, params, num_sites, num_states)

    # save true tree
    fname = os.path.join(path, 'true_tree')
    file = open(fname, 'wb+')
    pickle.dump(true_tree, file)
    
    # save params to recreate infinitesimal generator Q
    fname = os.path.join(path, 'params')
    file = open(fname, 'wb+')
    pickle.dump(params, file)
    
if __name__ == '__main__':
    simulate_data()