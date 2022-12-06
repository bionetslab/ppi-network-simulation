import pandas as pd
import itertools as itt
from scipy.stats import wasserstein_distance
from os.path import join
import networkx as nx
import ppinetsim.utils as utils
from ppinetsim.parameters import Parameters
from ppinetsim.simulator import run_simulation
import seaborn as sns


def estimate_likelihood(parameters: Parameters, num_simulations_per_generator=50, verbose=True):
    """

    Parameters
    ----------
    parameters : Parameters
      Specifies all parameters of the simulation.
    num_simulations_per_generator : `int`
      Numbers of simulations per generator.
    verbose : `bool`
      If True, a progress bar is displayed (does not work in Jupyter notebook).

    Returns
    -------
    likelihood_at_k : pd.DataFrame
      Data frame with likelihood k-NN estimate of the observed network's likelihood given a BA and an ER ground truth.
    """
    if not parameters.sample_studies:
        raise ValueError('Parameter "sample_studies" must be set to True for Bayesian inference.')
    adj_observed = _construct_observed_network(parameters)
    node_degrees_observed = utils.node_degrees(adj_observed)
    degree_dist_observed = utils.degrees_to_distribution(node_degrees_observed)
    generators = ['erdos-renyi', 'barabasi-albert']
    distances_from_observed = []
    for generator in generators:
        parameters.generator = generator
        for _ in range(num_simulations_per_generator):
            node_degrees_simulated, _, _, _ = run_simulation(parameters, verbose)
            degree_dist_simulated = utils.degrees_to_distribution(node_degrees_simulated)
            distance_from_aggregated = wasserstein_distance(degree_dist_observed[0, ], degree_dist_simulated[0, ],
                                                            degree_dist_observed[1, ], degree_dist_simulated[1, ])
            distances_from_observed.append((distance_from_aggregated, generator))
    distances_from_observed.sort()
    likelihood_at_k = pd.DataFrame(columns=['k', 'Erdos-Renyi', 'Barabasi-Albert'], dtype=float)
    erdos_renyi_count = 0
    barabasi_albert_count = 0
    k = 0
    for _, generator in distances_from_observed:
        k += 1
        if generator ==  'erdos-renyi':
            erdos_renyi_count += 1
        else:
            barabasi_albert_count += 1
        likelihood_at_k.loc[k-1, 'k'] = k
        likelihood_at_k.loc[k-1, 'Erdos-Renyi'] = erdos_renyi_count / k
        likelihood_at_k.loc[k-1, 'Barabasi-Albert'] = barabasi_albert_count / k
    return likelihood_at_k


def plot_likelihoods(likelihood_at_k, ax=None):
    data = likelihood_at_k.melt(value_vars=['Erdos-Renyi', 'Barabasi-Albert'], id_vars=['k'], var_name='Generator',
                                value_name='Likelihood')
    return sns.lineplot(data=data, x='k', y='likelihood', hue='generator', ax=ax)


def _construct_observed_network(parameters: Parameters):
    edge_list = []
    for sampled_study in parameters.sampled_studies:
        filename = join('ppinetsim', 'data', parameters.test_method, f'{sampled_study}.csv')
        adj_sampled_study = pd.read_csv(filename, index_col=0)
    for edge in itt.product(adj_sampled_study.index, adj_sampled_study.columns):
        if adj_sampled_study.loc[edge]:
            edge_list.append(edge)
    observed_network = nx.Graph()
    observed_network.add_edges_from(edge_list)
    return nx.to_numpy_array(observed_network, dtype=bool)
