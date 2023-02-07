import pandas as pd
import itertools as itt
from scipy.stats import wasserstein_distance
from os.path import join
import networkx as nx
import ppinetsim.utils as utils
from ppinetsim.parameters import Parameters
from ppinetsim.simulator import run_simulation
import seaborn as sns


def estimate_posteriors(parameters: Parameters, num_simulations_per_generator=10, verbose=True):
    """Estimates posterior probabilities of observed PPI network having emerged from PL-distributed or from binomially
    distributed ground-truth interactome.

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
    posterior_at_k : pd.DataFrame
      Data frame with k-NN estimates of the posterior probabilities that observed network has emerged from PL- or from
      binomially distributed ground truth.
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
    posterior_at_k = pd.DataFrame(columns=['k', 'Erdos-Renyi', 'Barabasi-Albert'], dtype=float)
    erdos_renyi_count = 0
    barabasi_albert_count = 0
    k = 0
    for _, generator, _ in all_results:
        k += 1
        if generator == 'erdos-renyi':
            erdos_renyi_count += 1
        else:
            barabasi_albert_count += 1
        posterior_at_k.loc[k-1, 'k'] = k
        posterior_at_k.loc[k-1, 'Ground truth binomially distributed'] = erdos_renyi_count / k
        posterior_at_k.loc[k-1, 'Ground truth PL-distributed'] = barabasi_albert_count / k
    return posterior_at_k, all_results


def plot_posteriors(posterior_at_k, ax=None):
    """Plots likelihoods.

    Parameters
    ----------
    posterior_at_k : pd.DataFrame
      Data frame of posteriors returned by `estimate_posteriors()`.
    ax : matplotlib.pyplot.axis
      Axis for the plot. If None, a new axis is generated.

    Returns
    -------
    matplotlib.pyplot.axis
      Axis containing the plot.

    """
    data = posterior_at_k.melt(value_vars=['Ground truth binomially distributed', 'Ground truth PL-distributed'],
                               id_vars=['k'], var_name='Class', value_name='Estimated posterior')
    return sns.lineplot(data=data, x='k', y='Estimated posterior', hue='Class',
                        hue_order=['Ground truth PL-distributed', 'Ground truth binomially distributed'], ax=ax)


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
