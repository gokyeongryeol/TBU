import logging

import numpy as np
from sklearn.metrics import pairwise_distances

from .strategy import Strategy


class KCenter(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_kcenter(self, u_emb, l_emb, n):
        dist = pairwise_distances(u_emb, l_emb)
        dist_min = dist.min(axis=1)

        k_idxs = []
        k_cand_idxs = list(range(len(u_emb)))
        for i in range(n):
            idx = dist_min.argmax()
            k_idxs.append(k_cand_idxs.pop(idx))

            if i < n - 1:
                k_emb = u_emb[[idx]]
                u_emb = np.delete(u_emb, idx, 0)

                dist_min = np.delete(dist_min, idx, 0)
                dist_idx = pairwise_distances(u_emb, k_emb).ravel()
                dist_min = np.stack([dist_min, dist_idx], axis=1).min(axis=1)

        k_idxs = np.array(k_idxs)

        return k_idxs

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
            u_emb = self.target.get_embedding(loader_unlab).numpy()

            dataset_lab = self.data.get_labeled_data()
            loader_lab = self.loader_fn(dataset_lab)
            l_emb = self.target.get_embedding(loader_lab).numpy()

            k_idxs = self.get_kcenter(u_emb, l_emb, n)
            q_idxs = m_idxs[k_idxs]

        elif m == n:
            logging.info(f"\t Option 2: Bounded({m}) = budget({n})")
            q_idxs = m_idxs

        else:
            logging.info(f"\t Option 3: Bounded({m}) < budget({n})")
            not_m_idxs = np.arange(U)[~m_sub_idxs]
            subset_unlab = self.data.get_indexed_data(dataset_unlab, not_m_idxs)
            loader_unlab = self.loader_fn(subset_unlab)
            u_emb = self.target.get_embedding(loader_unlab).numpy()

            dataset_lab = self.data.get_labeled_data()
            loader_lab = self.loader_fn(dataset_lab)
            l_emb = self.target.get_embedding(loader_lab).numpy()

            subset_add = self.data.get_indexed_data(dataset_unlab, m_idxs)
            loader_add = self.loader_fn(subset_add)
            a_emb = self.target.get_embedding(loader_add).numpy()

            k_idxs = self.get_kcenter(
                u_emb, np.concatenate([l_emb, a_emb], axis=0), n - m
            )
            q_idxs = np.concatenate([m_idxs, not_m_idxs[k_idxs]], axis=0)

        assert len(q_idxs) == n
        return idxs_unlab[m_idxs], idxs_unlab[q_idxs]
