import logging
import time

import torch

from active.dataset import prepare_data
from active.models import prepare_model, train
from active.selection import Badge, Entropy, KCenter, Random
from active.utils import get_random_seed, reset_random_seed, set_random_seed


def active_learn(args, p_args, t_args):
    p_accs, p_nlls = [], []
    t_accs, t_nlls = [], []
    for rd in range(1, args.n_round + 1):
        set_random_seed(args.seed)
        p_acc, p_nll, t_acc, t_nll = active_one_round(rd, args, p_args, t_args)

        if p_acc is not None:
            p_accs.append(p_acc)
            torch.save(p_accs, f"./saved/metric/{p_args.path}/acc.pt")

        if p_nll is not None:
            p_nlls.append(p_nll)
            torch.save(p_nlls, f"./saved/metric/{p_args.path}/nll.pt")

        t_accs.append(t_acc)
        torch.save(t_accs, f"./saved/metric/{t_args.path}/acc.pt")

        t_nlls.append(t_nll)
        torch.save(t_nlls, f"./saved/metric/{t_args.path}/nll.pt")


def active_one_round(rd, args, p_args, t_args):
    logging.info(f"Round [{rd}]")

    tic = time.time()
    data = prepare_data(rd, args)

    p_acc, p_nll = None, None
    if rd <= args.r_round:
        if (
            (rd == args.n_round)
            or (args.algorithm == "Random" and args.percentile < 0.0)
            or (not args.semi and (p_args.arc == t_args.arc))
        ):
            proxy = None
        else:
            proxy, p_acc, p_nll = prepare_model(rd, data, p_args)

        target, t_acc, t_nll = prepare_model(rd, data, t_args)
    else:
        if (
            (rd == args.n_round)
            or (args.algorithm == "Random" and args.percentile < 0.0)
            or (not args.semi and (p_args.arc == t_args.arc))
        ):
            proxy = None
        else:
            seeds = get_random_seed()
            proxy, p_acc, p_nll = train(rd, data, p_args)
            reset_random_seed(*seeds)
            torch.save(proxy.state_dict(), f"./saved/ckpt/{p_args.path}/round_{rd}.pt")

        target, t_acc, t_nll = train(rd, data, t_args)
        torch.save(target.state_dict(), f"./saved/ckpt/{t_args.path}/round_{rd}.pt")

    torch.cuda.empty_cache()
    toc = time.time()
    logging.info(f"\t Model training took {round(toc - tic, 4)} seconds \n")

    if args.r_round <= rd < args.n_round:
        tic = time.time()

        method = globals()[args.algorithm](
            data,
            proxy,
            target
            if args.percentile >= 0 or args.n_filter > 0 or (not args.semi and (p_args.arc == t_args.arc))
            else proxy,
            args.n_MCdrop,
            args.percentile,
            args.n_filter,
            batch_size=args.n_ev_batch,
            num_workers=args.n_worker,
        )
        t_idxs, q_idxs = method.query(args.n_query)

        torch.save(t_idxs, f"./saved/idxs/{args.path}/round_{rd}_candidates.pt")
        torch.save(q_idxs, f"./saved/idxs/{args.path}/round_{rd}.pt")

        toc = time.time()
        logging.info(f"\t Data selection took {round(toc - tic, 4)} seconds \n")

    return p_acc, p_nll, t_acc, t_nll
