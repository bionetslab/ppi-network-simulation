import pandas as pd
import itertools as itt
from scipy.stats import wasserstein_distance
from os.path import join
import networkx as nx
import ppinetsim.utils as utils
from ppinetsim.parameters import Parameters
from ppinetsim.simulator import run_simulation
import seaborn as sns


def estimate_likelihood(parameters: Parameters, num_simulations_per_generator=10, verbose=True):
    """Estimates likelihood of observed network given Barabasi-Albert and Erdos-Renyi models.

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
    all_results : list
      List 3-tuples containing the results. The first entry contains the EMD of the simulated network from the observed
      network, the second entry the generator used for the hypothetical ground truth, and the third entry the degree
      distribution of the simulated network.
    """
    if not parameters.sample_studies:
        raise ValueError('Parameter "sample_studies" must be set to True for Bayesian inference.')
    adj_observed = _construct_observed_network(parameters)
    node_degrees_observed = utils.node_degrees(adj_observed)
    degree_dist_observed = utils.degrees_to_distribution(node_degrees_observed)
    generators = ['erdos-renyi', 'barabasi-albert']
    all_results = []
    for generator in generators:
        parameters.generator = generator
        for _ in range(num_simulations_per_generator):
            node_degrees_simulated, _, _, _ = run_simulation(parameters, verbose)
            degree_dist_simulated = utils.degrees_to_distribution(node_degrees_simulated)
            distance_from_aggregated = wasserstein_distance(degree_dist_observed[0, ], degree_dist_simulated[0, ],
                                                            degree_dist_observed[1, ], degree_dist_simulated[1, ])
            all_results.append((distance_from_aggregated, generator, degree_dist_simulated))
    all_results.sort()
    likelihood_at_k = pd.DataFrame(columns=['k', 'Erdos-Renyi', 'Barabasi-Albert'], dtype=float)
    erdos_renyi_count = 0
    barabasi_albert_count = 0
    k = 0
    for _, generator, _ in all_results:
        k += 1
        if generator == 'erdos-renyi':
            erdos_renyi_count += 1
        else:
            barabasi_albert_count += 1
        likelihood_at_k.loc[k-1, 'k'] = k
        likelihood_at_k.loc[k-1, 'Erdos-Renyi'] = erdos_renyi_count / k
        likelihood_at_k.loc[k-1, 'Barabasi-Albert'] = barabasi_albert_count / k
    return likelihood_at_k, all_results


def plot_likelihoods(likelihood_at_k, ax=None):
    """Plots likelihoods.

    Parameters
    ----------
    likelihood_at_k : pd.DataFrame
      Data frame of likelihoods returned by `estimate_likelihood()`.
    ax : matplotlib.pyplot.axis
      Axis for the plot. If None, a new axis is generated.

    Returns
    -------
    matplotlib.pyplot.axis
      Axis containing the plot.

    """
    data = likelihood_at_k.melt(value_vars=['Erdos-Renyi', 'Barabasi-Albert'], id_vars=['k'], var_name='Generator',
                                value_name='Likelihood')
    return sns.lineplot(data=data, x='k', y='Likelihood', hue='Generator', hue_order=['Erdos-Renyi', 'Barabasi-Albert'],
                        ax=ax)


def plot_distances(all_results, kind='box', ax=None):
    """Plots earth mover's distances of simulated networks from observed network.

    Parameters
    ----------
    all_results : list
      Lists of results returned by `estimate_likelihood()'.
    kind : str
      Kind of the plot, either 'box' or 'violin'.
    ax : matplotlib.pyplot.axis

    Returns
    -------

    """
    generator_map = {'erdos-renyi': 'Erdos-Renyi', 'barabasi-albert': 'Barabasi-Albert'}
    data = pd.DataFrame(data={'Generator': [generator_map[generator] for _, generator, _ in all_results],
                              'EMD from observed network': [dist for dist, _, _ in all_results]})
    if kind == 'box':
        return sns.boxplot(data=data, x='Generator', y='EMD from observed network',
                           order=['Erdos-Renyi', 'Barabasi-Albert'], ax=ax)
    elif kind == 'violin':
        return sns.violinplot(data=data, x='Generator', y='EMD from observed network', cut=0,
                           order=['Erdos-Renyi', 'Barabasi-Albert'], ax=ax)
    else:
        raise RuntimeError(f'Invalid argument kind="{kind}". Valid options: "box", "violin".')


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
