import logging

import numpy as np
import torch
from scipy import stats

from .strategy import Strategy


def init_centers(X, K):
    embs = torch.Tensor(X)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    # embs = embs.cuda() # allow for gpu usage
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.0] * len(embs)
    cent = 0

    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
        else:
            newD = torch.cdist(mu[-1].view(1, -1), embs, 2)[0].cpu().numpy()
            for i in range(len(embs)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        assert sum(D2) != 0.0
        D2 = D2.ravel().astype(float)
        Ddist = (D2**2) / sum(D2**2)
        for ind in indsAll:
            Ddist[ind] = 0.0
        customDist = stats.rv_discrete(name="custm", values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll


class Badge(Strategy):
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

            u_grad = self.target.get_gradient_emb(loader_unlab, self.n_MCdrop).numpy()
            b_idxs = init_centers(u_grad, n)
            q_idxs = m_idxs[b_idxs]

        elif m == n:
            logging.info(f"\t Option 2: Bounded({m}) = budget({n})")
            q_idxs = m_idxs

        else:
            logging.info(f"\t Option 3: Bounded({m}) < budget({n})")
            U = len(dataset_unlab)
            not_m_idxs = np.arange(U)[~m_sub_idxs]
            subset_unlab = self.data.get_indexed_data(dataset_unlab, not_m_idxs)
            loader_unlab = self.loader_fn(subset_unlab)

            u_grad = self.target.get_gradient_emb(loader_unlab, self.n_MCdrop).numpy()
            b_idxs = init_centers(u_grad, n - m)
            q_idxs = np.concatenate([m_idxs, not_m_idxs[b_idxs]], axis=0)

        assert len(q_idxs) == n
        return idxs_unlab[m_idxs], idxs_unlab[q_idxs]
