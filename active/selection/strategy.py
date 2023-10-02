import logging

import torch
from torch.utils.data import DataLoader

from active.models.network import Laplace


class Strategy:
    def __init__(self, data, proxy, target, n_MCdrop, percentile, **kwargs):
        self.data = data
        self.proxy = proxy
        self.target = target
        self.n_MCdrop = n_MCdrop
        self.percentile = percentile

        self.model_fn = lambda model: Laplace(
            model.arc, model.dp_rate, model.n_classes, use_thres=False
        )
        self.loader_fn = lambda dataset: DataLoader(dataset, **kwargs)
        self.loader_fn_s = lambda dataset: DataLoader(dataset, shuffle=True, **kwargs)

    def _medium_constrain(self):
        dataset_lab = self.data.get_labeled_data()
        loader_lab = self.loader_fn_s(dataset_lab)
        dataset_valid = self.data.get_valid_data()
        loader_valid = self.loader_fn_s(dataset_valid)

        proxy = self.model_fn(self.proxy)
        proxy.cuda()
        proxy.load_state_dict(self.proxy.state_dict(), strict=False)
        proxy.get_optimal_mff(loader_lab, loader_valid)
        logging.info(f"\t Optimized mff : {proxy.mff.data.item()}")

        pred_lab = proxy.get_prediction(loader_lab, self.n_MCdrop)
        ent_lab = (-pred_lab * torch.log(pred_lab + 1e-10)).sum(dim=1)
        target_lab = torch.tensor(dataset_lab.targets)

        dataset_unlab = self.data.get_unlabeled_data()[1]
        loader_unlab = self.loader_fn(dataset_unlab)

        pred_unlab = proxy.get_prediction(loader_unlab, self.n_MCdrop)
        ent_unlab = (-pred_unlab * torch.log(pred_unlab + 1e-10)).sum(dim=1)
        target_unlab = pred_unlab.argmax(dim=1)

        U = len(dataset_unlab)
        ne_sub_idxs = torch.zeros(U, dtype=torch.bool)
        for c in range(proxy.n_classes):
            l_sub_idxs = target_lab == c
            u_sub_idxs = target_unlab == c

            bound = ent_lab[l_sub_idxs].quantile(q=self.percentile)
            c_sub_idxs = ent_unlab[u_sub_idxs].ge(bound)

            c_idxs = torch.arange(U)[u_sub_idxs][c_sub_idxs]
            ne_sub_idxs[c_idxs] = True

        nh_sub_idxs = (~self.proxy.hard_idxs).cpu()
        m_sub_idxs = ne_sub_idxs * nh_sub_idxs

        logging.info(
            f"\t Easy({sum(~ne_sub_idxs)}), Hard({sum(~nh_sub_idxs)}), Medium({sum(m_sub_idxs)})"
        )

        return m_sub_idxs

    def constrain(self):
        if self.percentile >= 0.0:
            return self._medium_constrain()
        else:
            dataset_unlab = self.data.get_unlabeled_data()[1]
            U = len(dataset_unlab)
            return torch.ones(U, dtype=torch.bool)
