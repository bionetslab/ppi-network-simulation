import json
import pandas as pd
from typing import Optional
from os.path import join
import numpy as np


class Parameters(object):
    """Parameter class.

    Attributes
    ----------
    num_proteins : `int`
      Number of proteins in ground-truth and simulated PPI network.
    num_ppis_ground_truth : `int`
      Number of PPIs in ground-truth PPI network.
    false_positive_rate : `float`
      False positive rate used during simulation. Range: [0,1]
    false_negative_rate : `float`
      False negative rate used during simulation. Range: [0,1].
    seed : `int` or None
      Seed used for randomizations.
    generator : `str`
      Choices: 'erdos-renyi', 'barabasi-albert'. Specifies random network model used to generate ground-truth.
    biased : `bool`
      If True, the observed PPI network is generated under study bias. Default: True.
    baseline_degree : `float`
      Specifies baseline value added to all degrees when generating PPI networks under study bias. Must be >0.
    test_method : `str`
      Choices: 'AP-MS', 'Y2H'. Specifies sampling strategy for protein pairs to be tested for interaction. If set to
      'AP-MS', only the bait proteins are sampled under study bias. If set to 'Y2H', both baits and preys are sampled
       under study bias. Has no effect if ``biased`` is set to False. Default: 'AP-MS'.
    num_baits : `str` or None
      Specifies maximum number of baits. If set to `None`, numbers of baits are sampled from real-world studies.
    num_preys : `int`
      Specifies maximum number of preys. If set to `None`, numbers of preys are sampled from real-world studies.
    acceptance_threshold : `float`
      Specifies lower bound on ratio of positive tests among all tests necessary for an edge to appear in the observed
      PPI network. If set to 0, one positive test is enough to ensure inclusion. Range: [0,1).
    num_studies : `int`
      Specifies maximum number of test rounds to be carried out during simulation. Automatically set to number of all
      studies contained in data/<test_method> directory if set to value <= 0. If sample_studies is True and num_studies
      is set to value <= number of all studies contained in data/<test_method> directory, studies are sampled without
      replacement. Otherwise, they are sampled with replacement.
    sample_studies : `bool`
      If True, numbers of baits and preys are sampled from real-world studies.
    sampled_studies : `list`
      If non-empty, numbers of baits and preys are sampled from provided list of studies.
    only_big_studies : `bool`
      If True, only studies with >= 200 PPIs are used for sampling numbers of baits and preys.
    pm_num_baits : `int`
      If set to value > 0, number of baits is uniformly sampled from the interval
      [max(1, num_baits - pm_num_baits), num_baits + pm_num_baits]. Default: 0.
    pm_num_preys : `int`
      If set to value > 0, number of preys is uniformly sampled from the interval
      [max(1, num_preys - pm_num_preys), num_preys + pm_num_preys]. Default: 0.
    """

    def __init__(self, path_to_json: Optional[str] = None):
        """Initialized Parameter object.

        Parameters
        ----------
        path_to_json : `str` or None
          Either None or path to JSON file specifying the attributes of the Parameter object. Attributes which are not
          provided in the JSON file are set to default values.
        """
        data = dict()
        if path_to_json:
            fp = open(path_to_json)
            data = json.load(fp)
            fp.close()
        self.num_proteins = data.get('num_proteins', 20000)
        self.num_ppis_ground_truth = data.get('num_ppis_ground_truth', 1000000)
        self.false_positive_rate = data.get('false_positive_rate', 0.2)
        self.false_negative_rate = data.get('false_negative_rate', 0.1)
        self.seed = data.get('seed', None)
        self.generator = data.get('generator', 'erdos-renyi')
        self.biased = bool(data.get('biased', True))
        self.baseline_degree = float(data.get('baseline_degree', 1.0))
        self.test_method = data.get('test_method', 'AP-MS')
        self.num_preys = data.get('num_preys', None)
        self.num_baits = data.get('num_baits', None)
        self.acceptance_threshold = float(data.get('acceptance_threshold', 0.0))
        self.num_studies = int(data.get('num_studies', 1000))
        self.sample_studies = bool(data.get('sample_studies', False))
        self.sampled_studies = data.get('sampled_studies', [])
        self.sample_studies = self.sample_studies or (self.num_preys is None or self.num_baits is None)
        self.sample_studies = self.sample_studies or (len(self.sampled_studies) > 0)
        self.only_big_studies = bool(data.get('only_big_studies', True))
        self.pm_num_baits = int(data.get('pm_num_baits', 0))
        self.pm_num_preys = int(data.get('pm_num_preys', 0))
        if self.sample_studies:
            rng = np.random.default_rng(self.seed)
            if self.only_big_studies:
                filename = join('ppinetsim', 'data', self.test_method, 'num_baits_preys_200.csv')
            else:
                filename = join('ppinetsim', 'data', self.test_method, 'num_baits_preys.csv')
            num_baits_preys = pd.read_csv(filename, index_col='study')
            if len(self.sampled_studies) == 0:
                all_studies = list(num_baits_preys.index)
                if self.num_studies <= 0:
                    self.sampled_studies = all_studies
                elif self.num_studies > len(all_studies):
                    self.sampled_studies = rng.choice(all_studies, size=self.num_studies, replace=True)
                else:
                    self.sampled_studies = rng.choice(all_studies, size=self.num_studies, replace=False)
            self.num_studies = len(self.sampled_studies)
            self.num_preys = []
            self.num_baits = []
            for sampled_study in self.sampled_studies:
                num_baits = num_baits_preys.loc[sampled_study, 'num_baits']
                num_preys = num_baits_preys.loc[sampled_study, 'num_preys']
                if self.pm_num_baits > 0:
                    num_baits = rng.integers(max(1, num_baits - self.pm_num_baits), num_baits + self.pm_num_baits + 1)
                if self.pm_num_preys > 0:
                    num_preys = rng.integers(max(1, num_preys - self.pm_num_preys), num_preys + self.pm_num_preys + 1)
                self.num_baits.append(num_baits)
                self.num_preys.append(num_preys)
