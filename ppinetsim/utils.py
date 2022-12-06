import networkx as nx
import numpy as np
from ppinetsim.parameters import Parameters


def initialize_matrices(parameters: Parameters):
    if parameters.generator.lower() == 'erdos-renyi':
        graph = nx.gnm_random_graph(parameters.num_proteins, parameters.num_ppis_ground_truth, seed=parameters.seed)
    elif parameters.generator.lower() == 'barabasi-albert':
        m = np.max([1, round(parameters.num_proteins / 2.0 - np.sqrt((parameters.num_proteins * parameters.num_proteins) / 4.0 - parameters.num_ppis_ground_truth))])
        graph = nx.barabasi_albert_graph(parameters.num_proteins, m, seed=parameters.seed)
    else:
        raise ValueError(f'Invalid generator name {parameters.generator}. Valid choices: "erdos-renyi", "barabasi-albert".')
    adj_ground_truth = nx.to_numpy_array(graph, dtype=bool)
    adj_simulated = np.ndarray((parameters.num_proteins, parameters.num_proteins), dtype=bool)
    num_tests = np.ndarray((parameters.num_proteins, parameters.num_proteins), dtype=np.int32)
    num_positive_tests = np.ndarray((parameters.num_proteins, parameters.num_proteins), dtype=np.int32)
    return adj_ground_truth, adj_simulated, num_tests, num_positive_tests


def sample_protein_pairs(adj_simulated: np.ndarray, parameters: Parameters, rng: np.random.Generator, i: int):
    num_proteins = adj_simulated.shape[0]
    if parameters.biased:
        p = node_degrees(adj_simulated, dtype=float) + parameters.baseline_degree
        p = p / p.sum()
    else:
        p = np.full(num_proteins, 1 / num_proteins)
    if parameters.sample_studies:
        num_preys = parameters.num_preys[i]
        num_baits = parameters.num_baits[i]
    else:
        num_preys = rng.integers(1, parameters.num_preys + 1)
        num_baits = rng.integers(1, parameters.num_baits + 1)
    baits = rng.choice(num_proteins, size=num_baits, replace=False, p=p)
    if parameters.test_method.upper() == 'AP-MS':
        preys = rng.choice(num_proteins, size=num_preys, replace=False)
    elif parameters.test_method.upper() == 'Y2H':
        preys = rng.choice(num_proteins, size=num_preys, replace=False, p=p)
    else:
        raise ValueError(f'Invalid test method name {parameters.test_method}. Valid choices: "AP-MS", "Y2H".')
    pairs = [(bait, prey) for bait in baits for prey in preys if bait != prey]
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


def update_simulated_ppi_network(protein_pairs: list, num_tests: np.ndarray, num_positive_tests: np.ndarray,
                                 adj_simulated: np.ndarray, parameters: Parameters):
    for u, v in protein_pairs:
        is_edge = ((num_positive_tests[u, v] / num_tests[u, v]) > parameters.acceptance_threshold)
        adj_simulated[u, v] = is_edge
        adj_simulated[v, u] = is_edge


def node_degrees(adj: np.ndarray, dtype=None):
    return np.squeeze(np.asarray(adj.sum(axis=0), dtype=dtype))


def num_edges(adj: np.ndarray):
    return int(adj.sum())


def degrees_to_frequencies(node_degrees: np.ndarray, dtype=None):
    return np.asarray(np.unique(node_degrees, return_counts=True), dtype=dtype)


def degrees_to_distribution(node_degrees: np.ndarray):
    freqs = degrees_to_frequencies(node_degrees, dtype=float)
    freqs[1, ] /= freqs[1, ].sum()
    return freqs
