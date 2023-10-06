from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_features(self, x):
        raise NotImplementedError

    @abstractmethod
    def forward_head(self, emb):
        raise NotImplementedError

    def forward(self, x):
        emb = self.forward_features(x)
        out = self.forward_head(emb)
        return out, emb

    def predict(self, x, n_MCdrop, pool):
        prob_lst = []
        emb = self.forward_features(x)
        for i in range(n_MCdrop):
            logit = self.forward_head(emb)
            prob_lst.append(F.softmax(logit, dim=-1))

        prob = torch.stack(prob_lst)
        if pool:
            prob = prob.mean(dim=0)
        return prob

    def get_embedding(self, loader):
        self.eval()

        D = len(loader.dataset)
        embDim = self.embDim
        embedding = torch.zeros([D, embDim])

        with torch.no_grad():
            for x, _, _, idxs in loader:
                emb = self.forward_features(x.cuda())
                embedding[idxs] = emb.cpu()

        return embedding

    def get_prediction(self, loader, n_MCdrop=1, use_strong=False, pool=True):
        self.eval()

        D = len(loader.dataset)
        C = self.n_classes
        prediction = torch.zeros([n_MCdrop, D, C])

        with torch.no_grad():
            for x1, x2, _, idxs in loader:
                x = x2 if use_strong else x1
                prob = self.predict(x.cuda(), n_MCdrop, pool=False)
                prediction[:, idxs] += prob.cpu()

        if pool:
            prediction = prediction.mean(dim=0)

        return prediction

    def get_gradient_emb(self, loader, n_MCdrop=1, use_label=False):
        self.eval()

        D = len(loader.dataset)
        W = self.linear.weight
        nLab, embDim = W.shape[0], W.shape[1]
        gradient = torch.zeros([D, nLab * embDim])

        with torch.no_grad():
            for i, (x, _, y, idxs) in enumerate(loader):
                x = x.cuda()
                emb = self.forward_features(x)
                prob = self.predict(x, n_MCdrop, pool=True)
                if use_label:
                    label = y.cuda()
                else:
                    label = torch.argmax(prob, dim=-1)

                B = len(idxs)
                conf = prob[range(B), label].unsqueeze(dim=1)

                emb, conf, label = emb.cpu(), conf.cpu(), label.cpu()
                common = (-emb * conf).repeat((1, nLab))
                mask = F.one_hot(label, nLab).repeat_interleave(embDim, 1)
                embs = emb.repeat((1, nLab))
                gradient[idxs] += common + mask * embs

        return gradient
