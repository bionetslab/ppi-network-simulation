import ppinetsim.utils as utils
from ppinetsim.parameters import Parameters
import numpy as np
from progress.bar import Bar


def run_simulation(parameters: Parameters, verbose=True):
    """Runs the simulation.

    Parameters
    ----------
    parameters : Parameters
      Specifies all parameters of the simulation.
    verbose : `bool`
      If True, a progress bar is displayed (does not work in Jupyter notebook).

    Returns
    -------
    node_degrees : numpy.ndarray
      2D array of node degrees in observed PPI network at different snapshots during simulation. Rows correspond
      to the snapshots, columns to proteins.
    num_ppis : numpy.ndarray
      1D array of numbers of PPIs in observed PPI network at different snapshots during simulation. Rows correspond to
      the same snapshots as rows in ``degree_distribution``.
    """
    adj_ground_truth, adj_observed, num_tests, num_positive_tests = utils.initialize_matrices(parameters)
    rng = np.random.default_rng()
    if verbose:
        bar = Bar('Simulation round', max=parameters.num_studies)
    for i in range(parameters.num_studies):
        protein_pairs = utils.sample_protein_pairs(adj_observed, parameters, rng, i)
        utils.test_protein_pairs(protein_pairs, adj_ground_truth, num_tests, num_positive_tests, parameters, rng)
        utils.update_observed_ppi_network(protein_pairs, num_tests, num_positive_tests, adj_observed, parameters)
        if verbose:
            bar.next()
    if verbose:
        bar.finish()
    node_degrees_observed = utils.node_degrees(adj_observed)
    num_ppis_observed = utils.num_edges(adj_observed)
    node_degrees_ground_truth = utils.node_degrees(adj_ground_truth)
    num_ppis_ground_truth = utils.num_edges(adj_ground_truth)
    return node_degrees_observed, num_ppis_observed, node_degrees_ground_truth, num_ppis_ground_truth
