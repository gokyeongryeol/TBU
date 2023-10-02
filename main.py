import argparse
import os
from copy import deepcopy

from active.app import active_learn
from active.utils import make_dirs, set_logger

SIZE = {
    1000: "small",
    2500: "medium",
    5000: "large",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=3061, help="random seed for initial labels"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="index of the gpu id",
    )
    parser.add_argument(
        "--export_id",
        type=str,
        default="cifar10",
        choices=[
            "cifar10",
            "cifar100",
            "svhn",
        ],
        help="the data to be explored",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="Badge",
        choices=[
            "Random",
            "KCenter",
            "Entropy",
            "Badge",
        ],
        help="the active learning algorithm to select subset",
    )

    parser.add_argument(
        "--n_worker", type=int, default=4, help="number of workers to load data"
    )
    parser.add_argument(
        "--n_ev_batch", type=int, default=256, help="evaluate batch size"
    )
    parser.add_argument(
        "--t_epoch",
        type=int,
        default=50,
        help="time interval between epochs to evaluate model",
    )

    parser.add_argument(
        "--r_round",
        type=int,
        default=0,
        help="the resume round of active learning",
    )
    parser.add_argument(
        "--n_round", type=int, default=5, help="number of rounds to proceed"
    )
    parser.add_argument(
        "--n_init", type=int, default=1000, help="number of initial labels"
    )
    parser.add_argument(
        "--n_query", type=int, default=1000, help="number of queries per round"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.1,
        help="percentile of data variance to be filtered",
    )
    parser.add_argument(
        "--n_filter",
        type=int,
        default=5,
        help="number of filterng step to update hard-level mask",
    )

    parser.add_argument(
        "--n_MCdrop",
        type=int,
        default=1,
        help="number of MC dropout samples",
    )
    parser.add_argument(
        "--dp_rate",
        default=0.0,
        type=float,
        help="dropout rate for the classifier",
    )

    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

    parser.add_argument(
        "--proxy_arc",
        type=str,
        choices=["res-18", "wrn-28-2"],
        help="the proxy architecture for AL training",
    )
    parser.add_argument(
        "--target_arc",
        type=str,
        choices=["res-18"],
        help="the target architecture for AL evaluation",
    )
    parser.add_argument(
        "--semi",
        action="store_true",
        help="whether to use semi-supervised learning",
    )

    args = parser.parse_args()
    net_dir = f"{args.proxy_arc}-{'semi' if args.semi else 'sup'}-{args.target_arc}-sup"
    algorithm = f"{args.algorithm}_{args.n_filter}_{args.percentile}_{args.seed}"
    args.size = f"{SIZE[args.n_init]}_{SIZE[args.n_query]}"

    args.path = f"{args.export_id}/{net_dir}/{algorithm}/{args.size}/"

    if args.export_id in ["cifar10", "svhn"]:
        args.n_classes = 10
    elif args.export_id == "cifar100":
        args.n_classes = 100
    else:
        raise NotImplementedError

    return args


def get_semisup_args(args):
    ss_args = deepcopy(args)

    assert (
        args.proxy_arc == "wrn-28-2"
    ), "Architecture of the proxy is constrained to be 'wrn-28-2'."
    ss_args.arc = "wrn-28-2"

    ss_args.mu = 7
    ss_args.n_tr_batch = 64
    ss_args.l2_reg = 5e-4

    if args.export_id in ["cifar100", "svhn"]:
        ss_args.w_lr = 0.1
        ss_args.w_epoch = 50
    else:
        ss_args.w_epoch = 0

    ss_args.n_epoch = 200
    ss_args.lr = 0.03
    ss_args.cycle = 7 / 16

    ss_args.semi = True
    ss_args.ema_m = 0.999
    ss_args.eta = 0.01
    ss_args.lamb = 1.0

    return ss_args


def get_sup_args(args, arc):
    s_args = deepcopy(args)

    s_args.arc = arc
    s_args.mu = 1
    s_args.n_tr_batch = 128

    s_args.l2_reg = 5e-4

    s_args.semi = False
    s_args.w_epoch = 0

    if arc == "wrn-28-2":
        s_args.n_epoch = 300

        s_args.lr = 0.1
        s_args.milestone = [160, 240, 280]
        s_args.gamma = 0.5
    elif arc == "res-18":
        s_args.n_epoch = 200

        if args.export_id == "svhn":
            s_args.lr = 0.01
            s_args.milestone = []
            s_args.gamma = 1.0
        else:
            s_args.lr = 0.1
            s_args.milestone = [160]
            s_args.gamma = 0.1
    else:
        raise NotImplementedError

    return s_args


if __name__ == "__main__":
    args = parse_args()
    if args.semi:
        p_args = get_semisup_args(args)
    else:
        p_args = get_sup_args(args, args.proxy_arc)

    t_args = get_sup_args(args, args.target_arc)

    p_args.path = os.path.join(args.path, "train")
    t_args.path = os.path.join(args.path, f"eval-{t_args.arc}")

    make_dirs(args, ["log", "idxs"])
    make_dirs(p_args, ["ckpt", "metric"])
    make_dirs(t_args, ["ckpt", "metric"])

    set_logger(args, record_args=True)

    active_learn(args, p_args, t_args)
