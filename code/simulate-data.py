import numpy as np
import os
import cassiopeia as cas

from cassiopeia.sim import BirthDeathFitnessSimulator
from cassiopeia.sim import Cas9LineageTracingDataSimulator
from util import extract_compact_Q

seed =10
rng = np.random.default_rng(seed)
path = '../data' # relative path to data directory

def simulate_evolutionary_tree():
    # instantiate BirthDeathFitnessSimulator() object and simulate tree
    bd_sim = cas.sim.BirthDeathFitnessSimulator(
        birth_waiting_distribution = lambda scale: rng.exponential(scale),
        initial_birth_scale = 0.5,
        death_waiting_distribution = lambda: rng.exponential(1.5),
        mutation_distribution = lambda: 1 if rng.uniform() < 0.5 else 0,
        fitness_distribution = lambda: rng.normal(0, .5),
        fitness_base = 1.3,
        num_extant = 400,
        random_seed=seed
    )
    true_tree = bd_sim.simulate_tree()
    
    # uncomment below to plot phylogenetic tree with plotly
    # fig = cas.pl.plot_plotly(true_tree, random_state=seed)
    # fig.show()
    
    return true_tree

def lineage_tracing(true_tree, params, num_sites, num_states):
   # instantiate Cas9 lineage tracing object & overlay data onto ground_truth_tree
    lt_sim = cas.sim.Cas9LineageTracingDataSimulator(
        number_of_cassettes = num_sites,
        size_of_cassette = 1,
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

def main():
    # define simulation hyper-parameters and parameters    
    num_sites = 40
    num_states = 50
    params = {'mutation_rate': np.repeat(0.1, num_sites), # mutation rates [λM_1, λM_2, ..., λM_NumSites]
              'deletion_rate': np.array([9e-4]), # deletion rate λD
              'transition_prob': {i: 1/num_states for i in range(num_states)}} # simplex P = [p_1 ... p_NumStates]
              # ^ probability p_i of transitioning from unedited state to mutated state i

    # run simulation
    true_tree = simulate_evolutionary_tree()
    lineage_tracing(true_tree, params, num_sites, num_states)
    
    # save compact Q, transition probabilties, and the character matrix
    Q_compact = extract_compact_Q(params['mutation_rate'], params['deletion_rate'])

    # save Q_compact, transition_prob, and true_tree.character_matrix
    # TODO: finish saving relevant files
    # fname = os.path.join(path, 'compact_Q')
    # np.savez_compressed()

if __name__ == '__main__':
    main()