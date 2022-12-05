import pandas as pd
from scipy.stats import wasserstein_distance
import ppinetsim.utils as utils
from ppinetsim.parameters import Parameters
from ppinetsim.simulator import run_simulation


def estimate_likelihood(parameters: Parameters, num_simulations_per_generator=50, verbose=True):
    if not parameters.sample_studies:
        raise ValueError('Parameter "sample_studies" must be set to True for Bayesian inference.')
    node_degrees_aggregated = _compute_node_degrees_in_aggregated_network(parameters)
    degree_dist_aggregated = utils.degrees_to_distribution(node_degrees_aggregated)
    generators = ['erdos-renyi', 'barabasi-albert']
    distances_from_aggregated = []
    for generator in generators:
        parameters.generator = generator
        for _ in range(num_simulations_per_generator):
            node_degrees_simulated, _, _, _ = run_simulation(parameters, verbose)
            degree_dist_simulated = utils.degrees_to_distribution(node_degrees_simulated)
            distance_from_aggregated = wasserstein_distance(degree_dist_aggregated[0, ], degree_dist_simulated[0, ],
                                                            degree_dist_aggregated[1, ], degree_dist_simulated[1, ])
            distances_from_aggregated.append((distance_from_aggregated, generator))
    distances_from_aggregated.sort()
    likelihood_at_k = pd.DataFrame(index=range(1, 2 * num_simulations_per_generator + 1), columns=generators, dtype=float)
    erdos_renyi_count = 0
    barabasi_albert_count = 0
    k = 0
    for _, generator in distances_from_aggregated:
        k += 1
        if generator == 'erdos-renyi':
            erdos_renyi_count += 1
        else:
            barabasi_albert_count += 1
        likelihood_at_k.loc[k, 'erdos-renyi'] = erdos_renyi_count / k
        likelihood_at_k.loc[k, 'barabasi-albert'] = barabasi_albert_count / k
    return likelihood_at_k


def _compute_node_degrees_in_aggregated_network(parameters: Parameters):
    return []
