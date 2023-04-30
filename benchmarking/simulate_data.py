from cassiopeia.sim import BirthDeathFitnessSimulator, Cas9LineageTracingDataSimulator
import numpy as np

def simulate_data(transition_prob=None,
                  mutation_rate=0.025,
                  deletion_rate=9e-4,
                  num_sites=40,
                  num_states=40, 
                  missing_data=0.2, 
                  exp_time=40,
                  num_extant=1000,
                  seed=0):
    
    # setup random num generator and transition probabilities
    rng = np.random.default_rng(seed)
    if transition_prob is None: # if no transition prob provided, assume uniform
        transition_prob = {i: 1/num_states for i in range(num_states)}
    
    # instantiate BirthDeathFitnessSimulator() object and simulate tree
    bd_sim = BirthDeathFitnessSimulator(
        birth_waiting_distribution = lambda scale: rng.exponential(scale),
        initial_birth_scale = 0.5,
        death_waiting_distribution = lambda: rng.exponential(1.5),
        mutation_distribution = lambda: 1 if rng.uniform() < 0.5 else 0,
        fitness_distribution = lambda: rng.normal(0, .5),
        fitness_base = 1.3,
        num_extant = num_extant,
        experiment_time = exp_time,
        random_seed=seed
    )
    true_tree = bd_sim.simulate_tree()
    
    # instantiate Cas9 lineage tracing object & overlay data onto ground_truth_tree
    lt_sim = Cas9LineageTracingDataSimulator(
        number_of_cassettes = 1,
        size_of_cassette = num_sites,
        mutation_rate = mutation_rate,
        state_generating_distribution = None,
        number_of_states = num_states,
        state_priors = transition_prob, # must be dict
        heritable_silencing_rate = deletion_rate,
        stochastic_silencing_rate = missing_data,
        heritable_missing_data_state = -1,
        stochastic_missing_data_state = -1,
        random_seed = seed
    )
    lt_sim.overlay_data(true_tree)
    
    # store params necessary for computing hyperbolic embeddings
    params = {'mutation_rate': np.repeat(mutation_rate, num_sites),     # mutation rates [λM_1, λM_2, ..., λM_NumSites]
              'deletion_rate': np.array([deletion_rate]),               # deletion rate λD
              'transition_prob': {i: transition_prob[i] for i in range(num_states)}}    # simplex P = [p_1 ... p_NumStates]
              # ^ probability p_i of transitioning from unedited state to mutated state i

    return true_tree, params