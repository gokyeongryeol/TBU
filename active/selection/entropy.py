import logging

import numpy as np
import torch

from .strategy import Strategy


class Entropy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def query(self, n):
        idxs_unlab, dataset_unlab = self.data.get_unlabeled_data()
        m_sub_idxs = self.constrain()

        U = len(dataset_unlab)
        m_idxs = np.arange(U)[m_sub_idxs]
        m = len(m_idxs)
        if m > n:
            logging.info(f"\t Option 1: Bounded({m}) > budget({n})")
            subset_unlab = self.data.get_indexed_data(dataset_unlab, m_idxs)
            loader_unlab = self.loader_fn(subset_unlab)

            e_prob = self.target.get_prediction(loader_unlab, self.n_MCdrop)
            e_ent = (-e_prob * torch.log(e_prob + 1e-10)).sum(dim=1)
            e_idxs = (-e_ent).sort()[1][:n]
            q_idxs = m_idxs[e_idxs]

        elif m == n:
            logging.info(f"\t Option 2: Bounded({m}) = budget({n})")
            q_idxs = m_idxs

        else:
            logging.info(f"\t Option 3: Bounded({m}) < budget({n})")
            not_m_idxs = np.arange(U)[~m_sub_idxs]
            subset_unlab = self.data.get_indexed_data(dataset_unlab, not_m_idxs)
            loader_unlab = self.loader_fn(subset_unlab)

            r_prob = self.target.get_prediction(loader_unlab, self.n_MCdrop)
            r_ent = (-r_prob * torch.log(r_prob + 1e-10)).sum(dim=1)
            r_idxs = (-r_ent).sort()[1][: n - m]

            q_idxs = np.concatenate([m_idxs, not_m_idxs[r_idxs]], axis=0)

        assert len(q_idxs) == n
        return idxs_unlab[m_idxs], idxs_unlab[q_idxs]
