import logging
import os
import random
from pprint import pformat

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_random_seed(rs):
    torch.manual_seed(rs)
    torch.cuda.manual_seed(rs)
    torch.cuda.manual_seed_all(rs)
    np.random.seed(rs)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(rs)


def get_random_seed():
    t = torch.random.get_rng_state()
    tc = torch.cuda.random.get_rng_state()
    tca = torch.cuda.random.get_rng_state_all()
    n = np.random.get_state()
    r = random.getstate()
    return t, tc, tca, n, r


def reset_random_seed(t, tc, tca, n, r):
    torch.random.set_rng_state(t)
    torch.cuda.random.set_rng_state(tc)
    torch.cuda.random.set_rng_state_all(tca)
    np.random.set_state(n)
    random.setstate(r)


def make_dirs(args, name_lst):
    path = args.path

    for name in name_lst:
        name_path = f"./saved/{name}/{path}"
        os.makedirs(name_path, exist_ok=True)


def set_logger(args, record_args):
    logging.basicConfig(
        format="%(message)s",
        filename=f"./saved/log/{args.path}/exp.log",
        filemode="w",
        level=logging.INFO,
    )

    if record_args:
        logging.info(pformat(vars(args)) + "\n")
