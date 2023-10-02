import torch


def compute_metric(nLab, true, prob):
    acc = (true == prob.max(dim=1)[1]).float().mean().item()
    nll = -torch.log(prob[range(len(prob)), true] + 1e-10).mean().item()

    return acc, nll


def ema_param_copy(model, ema_model):
    for param, param_ema in zip(
        model.state_dict().values(), ema_model.state_dict().values()
    ):
        param.data.copy_(param_ema.data)


def ema_param_update(model, ema_model, ema_m):
    for param, param_ema in zip(
        model.state_dict().values(), ema_model.state_dict().values()
    ):
        if param_ema.dtype == torch.float32:
            param_ema.mul_(ema_m)
            param_ema.add_(param * (1 - ema_m))


def ema_thres_update(thres, thres_ema, ema_m):
    thres = thres_ema * ema_m + thres * (1 - ema_m)
    return thres


def hard_masking(loader_sub, model, ema_model, n_MCdrop):
    model.eval()

    D = len(loader_sub.dataset)
    hard_idxs = torch.zeros(D, dtype=torch.bool).cuda()

    for u_b in loader_sub:
        u_x, _, _, idxs = u_b

        u_prob = ema_model.predict(u_x.cuda(), n_MCdrop=n_MCdrop, pool=True)

        max_prob, max_idx = torch.max(u_prob, dim=1)
        thres_g = model.ema_g
        thres = thres_g * model.ema_l / torch.max(model.ema_l)

        bound = thres[max_idx]
        h_sub_idxs = max_prob.lt(bound)

        hard_idxs[idxs] = h_sub_idxs

    return hard_idxs


def replace_inf_to_zero(val):
    val[val == float("inf")] = 0.0
    return val
