import networkx as nx
import numpy as np
from itertools import combinations
from ppinetsim.parameters import Parameters


def initialize_matrices(parameters: Parameters):
    if parameters.generator.lower() == 'erdos-renyi':
        graph = nx.gnm_random_graph(parameters.num_proteins, parameters.num_ppis_ground_truth, seed=parameters.seed)
    elif parameters.generator.lower() == 'barabasi-albert':
        m = np.max([1, round(parameters.num_proteins / 2.0 - np.sqrt((parameters.num_proteins * parameters.num_proteins) / 4.0 - parameters.num_ppis_ground_truth))])
        graph = nx.barabasi_albert_graph(parameters.num_proteins, m, seed=parameters.seed, initial_graph=None)
    else:
        raise ValueError(f'Invalid generator name {parameters.generator}. Valid choices: "erdos-renyi", "barabasi-albert".')
    adj_ground_truth = nx.to_numpy_array(graph, dtype=bool)
    adj_observed = np.ndarray((parameters.num_proteins, parameters.num_proteins), dtype=bool)
    num_tests = np.ndarray((parameters.num_proteins, parameters.num_proteins), dtype=np.int32)
    num_positive_tests = np.ndarray((parameters.num_proteins, parameters.num_proteins), dtype=np.int32)
    return adj_ground_truth, adj_observed, num_tests, num_positive_tests


def sample_protein_pairs(adj_observed: np.ndarray, parameters: Parameters, rng: np.random.Generator):
    num_proteins = adj_observed.shape[0]
    if parameters.biased:
        p = degree_distribution(adj_observed, dtype=float) + parameters.baseline_degree
        p = p / p.sum()
    else:
        p = np.full(num_proteins, 1 / num_proteins)
    if parameters.test_method.upper() == 'AP-MS':
        bait = rng.choice(num_proteins, p=p)
        pairs = [(bait, prey) for prey in range(num_proteins) if prey != bait]
    elif parameters.test_method.upper() == 'Y2H':
        proteins = rng.choice(num_proteins, size=parameters.matrix_size, replace=False, p=p)
        pairs = list(combinations(proteins, 2))
    else:
        raise ValueError(f'Invalid test method name {parameters.test_method}. Valid choices: "AP-MS", "Y2H".')
    return pairs


def test_protein_pairs(protein_pairs: list, adj_ground_truth: np.ndarray, num_tests: np.ndarray,
                       num_positive_tests: np.ndarray, parameters: Parameters, rng: np.random.Generator):
    random_numbers = rng.uniform(size=len(protein_pairs))
    i = 0
    for u, v in protein_pairs:
        num_tests[u, v] += 1
        num_tests[v, u] += 1
        if adj_ground_truth[u, v]:
            if random_numbers[i] > parameters.false_negative_rate:
                num_positive_tests[u, v] += 1
                num_positive_tests[v, u] += 1
        elif random_numbers[i] <= parameters.false_positive_rate:
            num_positive_tests[u, v] += 1
            num_positive_tests[v, u] += 1
        i += 1


def update_observed_ppi_network(protein_pairs: list, num_tests: np.ndarray, num_positive_tests: np.ndarray,
                                adj_observed: np.ndarray, parameters: Parameters):
    for u, v in protein_pairs:
        is_edge = ((num_positive_tests[u, v] / num_tests[u, v]) > parameters.acceptance_threshold)
        adj_observed[u, v] = is_edge
        adj_observed[v, u] = is_edge


def degree_distribution(adj, dtype=None):
    return np.squeeze(np.asarray(adj.sum(axis=0), dtype=dtype))


def num_edges(adj):
    return int(adj.sum())
