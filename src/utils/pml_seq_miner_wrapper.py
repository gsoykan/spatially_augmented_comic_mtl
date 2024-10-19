import itertools
import random
from collections import defaultdict

import torch
from pytorch_metric_learning import miners as pml_miners
from torch import nn


class PMLSeqMinerWrapper(nn.Module):

    def __init__(self, miner: pml_miners.BaseMiner):
        super().__init__()
        self.miner = miner

    def __call__(self,
                 embeddings: torch.Tensor,
                 char_ids: torch.Tensor,
                 seq_ids: torch.Tensor,
                 original_seq_ids: torch.Tensor,  # this is actually series id
                 mix_series: bool = False):
        # indices_tuple = self.miner(embeddings, char_ids)

        zipped = list(zip(list(range(len(embeddings))), char_ids, seq_ids, original_seq_ids))
        seqs_chars_group = itertools.groupby(sorted(zipped,
                                                    key=lambda x: (x[2].item(), x[3].item())),
                                             lambda x: (x[2].item(), x[3].item()))

        original_seqs_chars_dict = defaultdict(dict)
        for (seq_id, original_seq_id), group in seqs_chars_group:
            original_seqs_chars_dict[original_seq_id][seq_id] = list(group)

        indices_tuples = []
        series_ids = list(original_seqs_chars_dict.keys())
        for original_series_id, seqs_chars_dict in original_seqs_chars_dict.items():

            other_elements = []
            if mix_series:
                # pick a random element from random combination for each other series...
                other_series_ids = list(filter(lambda x: x != original_series_id, series_ids))
                for other_series_id in other_series_ids:
                    other_seq_chars_dict = original_seqs_chars_dict[other_series_id]
                    random_key = random.choice(list(other_seq_chars_dict.keys()))
                    other_elements.append(random.choice(other_seq_chars_dict[random_key]))

            for seq_id, v in seqs_chars_dict.items():
                v = other_elements + v
                curr_char_ids_list = list(map(lambda e: e[1], v))
                curr_char_ids = torch.stack(curr_char_ids_list)
                curr_idxs = list(map(lambda e: e[0], v))
                idxs_dict = {i: j for i, j in enumerate(curr_idxs)}
                curr_embeddings = embeddings[curr_idxs]
                curr_indices_tuple = self.miner(curr_embeddings, curr_char_ids)
                is_all_empty = all(torch.numel(tensor) == 0 for tensor in curr_indices_tuple)
                if is_all_empty:
                    continue
                # convert curr_indices_tuple to actual indices...
                updated_indices_tuple = []
                for indices in curr_indices_tuple:
                    updated_indices = torch.tensor(list(map(lambda x: idxs_dict[x.item()], indices))).to(indices.device)
                    updated_indices_tuple.append(updated_indices)
                indices_tuples.append(updated_indices_tuple)

        indices_tuples = [torch.cat(sublist, dim=0).long() for sublist in zip(*indices_tuples)]

        return indices_tuples
