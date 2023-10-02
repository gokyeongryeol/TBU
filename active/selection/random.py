import logging

import numpy as np

from .strategy import Strategy


class Random(Strategy):
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
            q_idxs = np.random.choice(m_idxs, n, replace=False)

        elif m == n:
            logging.info(f"\t Option 2: Bounded({m}) = budget({n})")
            q_idxs = m_idxs

        else:
            logging.info(f"\t Option 3: Bounded({m}) < budget({n})")
            not_m_idxs = np.arange(U)[~m_sub_idxs]

            r_idxs = np.random.choice(np.arange(U - m), n - m, replace=False)
            q_idxs = np.concatenate([m_idxs, not_m_idxs[r_idxs]], axis=0)

        assert len(q_idxs) == n
        return idxs_unlab[m_idxs], idxs_unlab[q_idxs]
