import ppinetsim.utils as utils
from ppinetsim.parameters import Parameters
import numpy as np
from progress.bar import Bar


def run_simulation(parameters: Parameters, verbose=False):
    """Runs the simulation.

    Parameters
    ----------
    parameters : Parameters
      Specifies all parameters of the simulation.
    verbose : `bool`
      If True, a progress bar is displayed (does not work in Jupyter notebook).

    Returns
    -------
    degree_distribution : numpy.ndarray
      2D array of degree distributions of observed PPI network at different snapshots during simulation. Rows correspond
      to the snapshots, columns to proteins.
    num_ppis : numpy.ndarray
      1D array of numbers of PPIs in observed PPI network at different snapshots during simulation. Rows correspond to
      the same snapshots as rows in ``degree_distribution``.
    """
    adj_ground_truth, adj_observed, num_tests, num_positive_tests = utils.initialize_matrices(parameters)
    rng = np.random.default_rng(parameters.seed)
    degree_distributions = []
    num_ppis = []
    if verbose:
        bar = Bar('Simulation round', max=parameters.max_num_tests)
    for i in range(parameters.max_num_tests):
        protein_pairs = utils.sample_protein_pairs(adj_observed, parameters, rng)
        utils.test_protein_pairs(protein_pairs, adj_ground_truth, num_tests, num_positive_tests, parameters, rng)
        utils.update_observed_ppi_network(protein_pairs, num_tests, num_positive_tests, adj_observed, parameters)
        early_exit = utils.num_edges(adj_observed) >= parameters.max_num_ppis_observed
        if early_exit or i == parameters.max_num_tests - 1 or i % parameters.degree_inspection_interval == 0:
            degree_distributions.append(utils.degree_distribution(adj_observed))
            num_ppis.append(utils.num_edges(adj_observed))
        if verbose:
            bar.next()
        if early_exit:
            break
    if verbose:
        bar.finish()
    degree_distributions = np.array(degree_distributions)
    num_ppis = np.array(num_ppis)
    return degree_distributions, num_ppis
