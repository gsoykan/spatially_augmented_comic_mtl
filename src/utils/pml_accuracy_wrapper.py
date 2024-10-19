import itertools
import logging
import math
import warnings
from collections import defaultdict

import faiss
import numpy as np
import torch
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import FaissKMeans
from torch import nn


class PMLAccuracyWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        logging.basicConfig()
        logging.getLogger(c_f.LOGGER_NAME).setLevel(logging.CRITICAL)
        # warnings.filterwarnings("ignore")
        # faiss.logger.setLevel('ERROR')
        self.calculator = PMLAccuracyWrapper.get_calculator()
        self.all_results = []

    @staticmethod
    def get_calculator():
        return AccuracyCalculator(include=(),
                                  exclude=(),
                                  avg_of_avgs=True,
                                  return_per_class=False,
                                  k=None,
                                  label_comparison_fn=None,
                                  device=None,
                                  knn_func=None,
                                  kmeans_func=FaissKMeans(niter=20,
                                                          gpu=True,
                                                          min_points_per_centroid=1,
                                                          max_points_per_centroid=10000000))

    def __call__(self,
                 embeddings: torch.Tensor,
                 char_ids: torch.Tensor,
                 seq_ids: torch.Tensor):
        zipped = list(zip(list(range(len(embeddings))), char_ids, seq_ids))
        seqs_chars_group = itertools.groupby(sorted(zipped, key=lambda x: x[2].item()), lambda x: x[2].item())
        seqs_chars_dict = {}
        for key, group in seqs_chars_group:
            seqs_chars_dict[key] = list(group)

        curr_results = []
        for seq_id, v in seqs_chars_dict.items():
            curr_char_ids = torch.stack(list(map(lambda e: e[1], v)))
            curr_idxs = list(map(lambda e: e[0], v))
            if len(curr_idxs) < 5:
                continue
            curr_embeddings = embeddings[curr_idxs]
            query = curr_embeddings
            query_labels = curr_char_ids
            result = self.calculator.get_accuracy(query, query_labels)
            has_nan = any(math.isnan(value) for value in result.values())
            if has_nan:
                continue
            curr_results.append(result)
        self.all_results.extend(curr_results)
        return self._avg_results(curr_results)

    def compute(self):
        return self._avg_results(self.all_results)

    def _avg_results(self, results):
        merged_res = defaultdict(list)
        for result in results:
            for k, v in result.items():
                merged_res[k] = [*merged_res[k], v]
        averages = {}
        for key, values in merged_res.items():
            averages[key] = np.mean(values)
        return averages

    def reset(self):
        self.all_results = []
