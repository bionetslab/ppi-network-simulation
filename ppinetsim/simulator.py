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
    node_degrees_simulated : numpy.ndarray
      2D array of node degrees in simulated PPI network at different snapshots during simulation. Rows correspond
      to the snapshots, columns to proteins.
    num_ppis_simulated : numpy.ndarray
      1D array of numbers of PPIs in simulated PPI network at different snapshots during simulation. Rows correspond to
      the same snapshots as rows in ``degree_distribution``.
    node_degrees_ground_truth : numpy.ndarray
      2D array of node degrees in ground-truth PPI network at different snapshots during simulation. Rows correspond
      to the snapshots, columns to proteins.
    num_ppis_ground_truth : numpy.ndarray
      1D array of numbers of PPIs in ground-truth PPI network at different snapshots during simulation. Rows correspond to
      the same snapshots as rows in ``degree_distribution``.
    """
    adj_ground_truth, adj_simulated, num_tests, num_positive_tests = utils.initialize_matrices(parameters)
    rng = np.random.default_rng()
    if verbose:
        bar = Bar('Simulation round', max=parameters.num_studies)
    for i in range(parameters.num_studies):
        protein_pairs = utils.sample_protein_pairs(adj_simulated, parameters, rng, i)
        utils.test_protein_pairs(protein_pairs, adj_ground_truth, num_tests, num_positive_tests, parameters, rng)
        utils.update_simulated_ppi_network(protein_pairs, num_tests, num_positive_tests, adj_simulated, parameters)
        if verbose:
            bar.next()
    if verbose:
        bar.finish()
    node_degrees_simulated = utils.node_degrees(adj_simulated)
    num_ppis_simulated = utils.num_edges(adj_simulated)
    node_degrees_ground_truth = utils.node_degrees(adj_ground_truth)
    num_ppis_ground_truth = utils.num_edges(adj_ground_truth)
    return node_degrees_simulated, num_ppis_simulated, node_degrees_ground_truth, num_ppis_ground_truth
