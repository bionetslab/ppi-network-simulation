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
        posterior_at_k.loc[k-1, 'Erdos-Renyi'] = erdos_renyi_count / k
        posterior_at_k.loc[k-1, 'Barabasi-Albert'] = barabasi_albert_count / k
    return posterior_at_k, all_results


def plot_posteriors(posteriors, parameters=None, ax=None):
    """Plots likelihoods.

    Parameters
    ----------
    posteriors : pd.DataFrame
      Data frame of posteriors returned by `estimate_posteriors()`.
    parameters : dict
      None or dictionary with keys 'test_method', 'false_positive_rate', 'false_negative_rate', and
      'acceptance_threshold'. If provided, the parameters are displayed in title of the plot.
    ax : matplotlib.pyplot.axis
      Axis for the plot. If None, a new axis is generated.

    Returns
    -------
    matplotlib.pyplot.axis
      Axis containing the plot.

    """
    posteriors.rename(columns={'Erdos-Renyi': 'Ground truth binomially distributed',
                               'Barabasi-Albert': 'Ground truth PL-distributed'}, inplace=True)
    data = posteriors.melt(value_vars=['Ground truth binomially distributed', 'Ground truth PL-distributed'],
                           id_vars=['k'], var_name='Class', value_name='Estimated posterior')
    return sns.lineplot(data=data, x='k', y='Estimated posterior', hue='Class',
                        hue_order=['Ground truth PL-distributed', 'Ground truth binomially distributed'],
                        ax=ax).set(title=_title(parameters))


def plot_distances(all_results, parameters=None, kind='box', ax=None):
    """Plots earth mover's distances of simulated networks from observed network.

    Parameters
    ----------
    all_results : list
      Lists of results returned by `estimate_likelihood()'.
    parameters : dict
      None or dictionary with keys 'test_method', 'false_positive_rate', 'false_negative_rate', and
      'acceptance_threshold'. If provided, the parameters are displayed in title of the plot.
    kind : str
      Kind of the plot, either 'box', 'violin', or 'swarm'.
    ax : matplotlib.pyplot.axis

    Returns
    -------

    """
    generator_map = {'erdos-renyi': 'Binomially distributed', 'barabasi-albert': 'PL-distributed'}
    data = pd.DataFrame(data={'Ground truth': [generator_map[generator] for _, generator, _ in all_results],
                              'EMD from observed network': [dist for dist, _, _ in all_results]})
    if kind == 'box':
        return sns.boxplot(data=data, x='Ground truth', y='EMD from observed network',
                           order=['PL-distributed', 'Binomially distributed'], ax=ax).set(title=_title(parameters))
    elif kind == 'violin':
        return sns.violinplot(data=data, x='Ground truth', y='EMD from observed network', cut=0,
                              order=['PL-distributed', 'Binomially distributed'], ax=ax).set(title=_title(parameters))
    elif kind == 'swarm':
        return sns.swarmplot(data=data, x='Ground truth', y='EMD from observed network',
                             order=['PL-distributed', 'Binomially distributed'], ax=ax).set(title=_title(parameters))
    else:
        raise RuntimeError(f'Invalid argument kind="{kind}". Valid options: "box", "violin", "swarm".')


def _title(parameters):
    if parameters is None:
        return ''
    title_parts = []
    if 'test_method' in parameters:
        title_parts.append(f'{parameters["test_method"]} testing')
    if 'false_positive_rate' in parameters:
        fpr = r'\mathit{FPR}'
        title_parts.append(r'${}={}$'.format(fpr, parameters['false_positive_rate']))
    if 'false_negative_rate' in parameters:
        fnr = r'\mathit{FNR}'
        title_parts.append(r'${}={}$'.format(fnr, parameters['false_negative_rate']))
    if 'acceptance_threshold' in parameters:
        title_parts.append(r'$\gamma={}$'.format(parameters['acceptance_threshold']))
    return ', '.join(title_parts)


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
