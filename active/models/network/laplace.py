import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .arc import Architecture

EPS = 1e-7


class Laplace(Architecture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        W = self.linear.weight
        device = W.device
        E = W.shape[1]

        self.register_buffer("precision", torch.zeros((E, E), device=device))
        self.register_buffer("covariance", torch.eye(E, device=device))
        self.mff = nn.Parameter(EPS * torch.ones(1, device=device))

    def reset_stat(self):
        device = self.precision.device

        self.precision = torch.zeros(self.precision.shape, device=device)
        self.covariance = torch.eye(self.precision.shape[1], device=device)
        self.mff.data = EPS * torch.ones(1, device=device)

    def update_prec(self, emb, conf):
        multiplier = conf * (1 - conf)
        minibatch = emb.t() @ (multiplier * emb)
        self.precision += minibatch

    def mean_field_approx(self, diag):
        scale = torch.sqrt(1.0 + diag * self.mff)
        scale = scale.unsqueeze(dim=1)
        return scale

    @torch.no_grad()
    def compute_feature_covar(self, loader):
        self.eval()

        self.reset_stat()
        for x, _, _, _ in loader:
            emb = self.forward_features(x.cuda())
            logit = self.forward_head(emb)

            prob = F.softmax(logit, dim=1)
            conf = prob.max(dim=1, keepdim=True)[0]
            self.update_prec(emb, conf)

        jitter = EPS * torch.eye(
            self.precision.shape[1],
            device=self.precision.device,
        )

        def inverse(precision):
            u, info = torch.linalg.cholesky_ex(precision + jitter)
            assert (info == 0).all(), "Precision matrix inversion failed!"
            return torch.cholesky_inverse(u)

        self.covariance = inverse(self.precision).contiguous()

    @torch.no_grad()
    def get_listed_values(self, loader_lab, loader_valid):
        self.compute_feature_covar(loader_lab)

        diag_lst, logit_lst, label_lst = [], [], []
        for x, _, y, _ in loader_valid:
            emb = self.forward_features(x.cuda())
            logit = self.forward_head(emb)

            logit_covariance = emb @ (self.covariance @ emb.t())
            diag = logit_covariance.diagonal()

            diag_lst.append(diag.cpu())
            logit_lst.append(logit.cpu())
            label_lst.append(y)

        return diag_lst, logit_lst, label_lst

    def get_optimal_mff(self, loader_lab, loader_valid):
        diag_lst, logit_lst, label_lst = self.get_listed_values(
            loader_lab, loader_valid
        )

        nll_criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.LBFGS(
            [self.mff], lr=0.01, max_iter=1000, tolerance_change=EPS
        )

        bias = self.linear.bias.data
        B = loader_valid.batch_size

        def eval():
            self.mff.data.copy_(torch.clamp(self.mff, min=EPS))
            optimizer.zero_grad()

            loss = 0.0
            for diag, logit, label in zip(diag_lst, logit_lst, label_lst):
                diag, logit, label = diag.cuda(), logit.cuda(), label.cuda()
                scale = self.mean_field_approx(diag)
                logit_s = (logit - bias) / scale + bias

                weight = len(label) / B
                loss = loss + weight * nll_criterion(logit_s, label)

            loss.backward()
            return loss

        optimizer.step(eval)
        self.mff.data.copy_(torch.clamp(self.mff, min=EPS))

    def get_entropy(self, loader, n_MCdrop):
        self.eval()

        D = len(loader.dataset)
        entropy = torch.zeros([D])

        with torch.no_grad():
            for x, _, _, idxs in loader:
                prob = self.predict(x.cuda(), n_MCdrop, pool=True)
                entropy[idxs] += (-prob * torch.log(prob)).sum(dim=1).cpu()

        return entropy

    def predict(self, x, n_MCdrop, pool):
        bias = self.linear.bias.data

        prob_lst = []
        emb = self.forward_features(x)
        for i in range(n_MCdrop):
            logit = self.forward_head(emb)

            logit_covariance = emb @ (self.covariance @ emb.t())
            diag = logit_covariance.diagonal()
            scale = self.mean_field_approx(diag)
            logit_s = (logit - bias) / scale + bias

            prob_lst.append(F.softmax(logit_s, dim=-1))

        prob = torch.stack(prob_lst)
        if pool:
            prob = prob.mean(dim=0)
        return prob
