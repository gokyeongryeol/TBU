import logging
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader

from .train_utils import (
    compute_metric,
    ema_param_update,
    ema_thres_update,
    hard_masking,
    replace_inf_to_zero,
)


def make_loader(dataset_lab, dataset_unlab, dataset_valid, args):
    batch_size = args.n_tr_batch

    loader_lab = DataLoader(
        dataset_lab,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
        num_workers=1 if args.semi else args.n_worker,
    )

    loader_unlab, loader_sub = None, None
    if args.semi:
        loader_unlab = DataLoader(
            dataset_unlab,
            shuffle=True,
            batch_size=args.mu * batch_size,
            drop_last=True,
            num_workers=args.n_worker,
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)

        loader_sub = DataLoader(
            dataset_unlab,
            shuffle=False,
            batch_size=args.n_ev_batch,
            num_workers=args.n_worker,
            worker_init_fn=seed_worker,
            generator=g,
        )

    loader_valid = DataLoader(
        dataset_valid,
        shuffle=False,
        batch_size=args.n_ev_batch,
        num_workers=args.n_worker,
    )

    return loader_lab, loader_unlab, loader_sub, loader_valid


def make_scheduler(optimizer, args):
    if args.semi:

        def lr_lambda(curr_batch):
            if curr_batch < args.w_epoch:
                return args.w_lr / args.lr
            else:
                ratio = (curr_batch - args.w_epoch) / args.n_batch
                lr = math.cos(math.pi * args.cycle * ratio)
                return lr

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
    else:
        scheduler = MultiStepLR(
            optimizer,
            milestones=args.milestone,
            gamma=args.gamma,
        )
    return scheduler


def train_model(
    rd, dataset_lab, dataset_unlab, dataset_valid, model, ema_model, optimizer, args
):
    loader_lab, loader_unlab, loader_sub, loader_valid = make_loader(
        dataset_lab,
        dataset_unlab,
        dataset_valid,
        args,
    )

    args.n_batch = len(loader_lab) * (args.n_epoch - args.w_epoch)
    scheduler = make_scheduler(optimizer, args)

    loss_fn = nn.CrossEntropyLoss().cuda()

    n_epoch = args.n_epoch
    for epoch in range(n_epoch):
        if args.semi and epoch >= args.w_epoch:
            tr_acc = semisup_one_epoch(
                epoch,
                loader_lab,
                loader_unlab,
                model,
                ema_model,
                optimizer,
                scheduler,
                loss_fn,
                args,
            )
            if args.n_filter > 0:
                e_epoch = args.n_epoch // 8
                q_epoch = args.n_epoch - args.n_filter * e_epoch
                if (epoch + 1) > q_epoch and (epoch + 1) % e_epoch == 0:
                    with torch.no_grad():
                        hard_idxs = hard_masking(
                            loader_sub, model, ema_model, args.n_MCdrop
                        )

                    current_cnt = sum(hard_idxs).item()
                    ema_model.hard_idxs.data *= hard_idxs
                    total_cnt = sum(ema_model.hard_idxs).item()

                    logging.info(
                        f"\t Epoch [{epoch + 1}], (Number of Hard) current: {current_cnt}, total: {total_cnt}"
                    )

        else:
            tr_acc = sup_one_epoch(
                epoch,
                loader_lab,
                model,
                optimizer,
                scheduler,
                loss_fn,
                args,
            )

        if (epoch + 1) % args.t_epoch == 0:
            logging.info(f"\t Epoch [{epoch + 1}], Training ACC [{round(tr_acc, 3)}]")

            ev_model = ema_model if args.semi else model
            va_acc, va_nll = evaluate(loader_valid, ev_model, args)

    return ev_model, va_acc, va_nll


def semisup_one_epoch(
    epoch,
    loader_lab,
    loader_unlab,
    model,
    ema_model,
    optimizer,
    scheduler,
    loss_fn,
    args,
):
    model.train()

    B = loader_lab.batch_size
    C = args.n_classes

    n_correct = 0.0
    iterator_unlab = iter(loader_unlab)
    for i, l_b in enumerate(loader_lab):
        l_x, _, l_y, _ = l_b
        l_x, l_y = l_x.cuda(), l_y.cuda()

        l_out = model(l_x)[0]
        loss = loss_fn(l_out, l_y)

        n_correct += (l_y == l_out.max(dim=1)[1]).sum().item()

        try:
            u_b = next(iterator_unlab)
        except StopIteration:
            iterator_unlab = iter(loader_unlab)
            u_b = next(iterator_unlab)

        u_x1, u_x2, _, _ = u_b
        u_x1, u_x2 = u_x1.cuda(), u_x2.cuda()

        with torch.no_grad():
            u_prob = model.predict(u_x1, n_MCdrop=args.n_MCdrop, pool=True)

        max_prob, max_idx = torch.max(u_prob, dim=1)

        prob_g = max_prob.mean()
        prob_l = u_prob.mean(dim=0)
        prob_h = torch.bincount(max_idx, minlength=C) / len(max_idx)

        model.ema_g = ema_thres_update(prob_g, model.ema_g, args.ema_m)
        model.ema_l = ema_thres_update(prob_l, model.ema_l, args.ema_m)
        model.ema_h = ema_thres_update(prob_h, model.ema_h, args.ema_m)

        p_tilt, h_tilt = model.ema_l, model.ema_h
        thres = model.ema_g * model.ema_l / torch.max(model.ema_l)

        mask = max_prob.ge(thres[max_idx])
        u_x, u_y = u_x2[mask], max_idx[mask]

        if len(u_y) > 0:
            u_out = model(u_x)[0]
            u_loss = loss_fn(u_out, u_y) * len(u_y) / (args.mu * B)
            loss = loss + args.lamb * u_loss

            tilt_prob = p_tilt * replace_inf_to_zero(1 / h_tilt)
            tilt_prob = tilt_prob / tilt_prob.sum()

            p_bar = F.softmax(u_out, dim=1).mean(dim=0)
            u_idx = torch.argmax(u_out, dim=1)
            h_bar = torch.bincount(u_idx, minlength=C) / len(u_idx)
            bar_prob = p_bar * replace_inf_to_zero(1 / h_bar)
            bar_prob = bar_prob / bar_prob.sum()

            h_loss = (tilt_prob * torch.log(bar_prob + 1e-10)).sum()
            loss = loss + args.eta * h_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.zero_grad()
        with torch.no_grad():
            ema_param_update(model, ema_model, args.ema_m)

    acc = n_correct / len(loader_lab.dataset)

    return acc


def sup_one_epoch(epoch, loader, model, optimizer, scheduler, loss_fn, args):
    model.train()

    n_correct = 0.0
    for x, _, y, _ in loader:
        l_x, l_y = x.cuda(), y.cuda()
        l_out = model(l_x)[0]
        l_prob = F.softmax(l_out, dim=-1)

        n_correct += (l_y == l_prob.max(dim=1)[1]).sum().item()
        loss = loss_fn(l_out, l_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    acc = n_correct / len(loader.dataset)
    return acc


def evaluate(loader, model, args):
    true = torch.tensor(loader.dataset.targets)
    prob = model.get_prediction(loader, args.n_MCdrop)
    acc, nll = compute_metric(args.n_classes, true, prob)
    logging.info(f"\t Evaluation ACC [{round(acc, 3)}], NLL [{round(nll, 3)}]")

    return acc, nll
