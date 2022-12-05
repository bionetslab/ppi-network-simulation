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
      If True, the observed PPI network is generated under study bias.
    baseline_degree : `float`
      Specifies baseline value added to all degrees when generating PPI networks under study bias. Must be >0.
    test_method : `str`
      Choices: 'AP-MS', 'Y2H'. Specifies sampling strategy for protein pairs to be tested for interaction. If set to
      'AP-MS', one bait protein is sampled and then tested against all other proteins. If set to 'Y2H', ``matrix_size``
      many proteins are randomly sampled and all unordered pairs are tested for interaction.
    num_baits : `str` or None
      Specifies number of proteins to be sampled for pairwise testing against the bait protein if ``test_method`` is set
       to 'AP-MS'. If set to `None`, all proteins are tested as preys. Must be >=1.
    num_preys : `int`
      Specifies number of proteins to be sampled for pairwise testing if ``test_method`` is set to 'Y2H'. Must be >=2.
    acceptance_threshold : `float`
      Specifies lower bound on ratio of positive tests among all tests necessary for an edge to appear in the observed
      PPI network. If set to 0, one positive test is enough to ensure inclusion. Range: [0,1).
    num_studies : `int`
      Specifies maximum number of test rounds to be carried out during simulation.
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
        self.num_preys = int(data.get('num_preys', None))
        self.num_baits = int(data.get('num_baits', None))
        self.acceptance_threshold = float(data.get('acceptance_threshold', 0.0))
        self.num_studies = int(data.get('num_studies', 1000))
        self.sample_studies = bool(data.get('sample_studies', False))
        self.sample_studies = self.sample_studies or (self.num_preys is None or self.num_baits is None)
        self.sampled_studies = []
        if self.sample_studies:
            filename = join('data', self.test_method, 'num_preys_baits.csv')
            rng = np.random.default_rng(self.seed)
            num_baits_preys = pd.read_csv(filename)
            all_studies = list(num_baits_preys.index)
            self.sampled_studies = rng.choice(all_studies, size=self.num_studies, replace=False)
            self.num_preys = []
            self.num_baits = []
            for sampled_study in self.sample_studies:
                self.num_preys.append(num_baits_preys.loc[sampled_study, 'num_preys'])
                self.num_baits.append(num_baits_preys.loc[sampled_study, 'num_baits'])
