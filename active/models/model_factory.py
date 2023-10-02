import torch
from torch.utils.data import DataLoader

from active.dataset import get_dataset_train, get_dataset_valid

from .network import Architecture
from .train_utils import ema_param_copy
from .trainer import evaluate, train_model

N_TRAIN = {"cifar10": 50000, "cifar100": 50000, "svhn": 73257}


def prepare_model(rd, data, args):
    model = make_model(rd, args)

    ckpt_path = f"./saved/ckpt/{args.path}/round_{rd}.pt"
    model.load_state_dict(torch.load(ckpt_path))

    dataset_valid = get_dataset_valid(data)
    loader_eval = DataLoader(
        dataset_valid,
        shuffle=False,
        batch_size=args.n_ev_batch,
        num_workers=args.n_worker,
    )

    ev_acc, ev_nll = evaluate(loader_eval, model, args)
    return model, ev_acc, ev_nll


def make_model(rd, args):
    torch.cuda.set_device(args.gpu_id)

    unlabel_size = N_TRAIN[args.export_id] - args.n_init - args.n_query * (rd - 1)

    model = Architecture(
        arc=args.arc,
        n_classes=args.n_classes,
        dp_rate=args.dp_rate,
        use_thres=args.semi,
        unlabel_size=unlabel_size,
        n_filter=args.n_filter,
    )

    model.cuda()
    return model


def make_ema_model(rd, model, args):
    ema_model = make_model(rd, args)
    ema_param_copy(model, ema_model)
    ema_model.eval()
    return ema_model


def make_optimizer(model, args):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.l2_reg,
        momentum=args.momentum,
        nesterov=True,
    )
    return optimizer


def train(rd, data, args):
    dataset_lab, dataset_unlab = get_dataset_train(data, args)
    dataset_valid = get_dataset_valid(data)

    model = make_model(rd, args)
    ema_model = make_ema_model(rd, model, args) if args.semi else None

    optimizer = make_optimizer(model, args)
    return train_model(
        rd, dataset_lab, dataset_unlab, dataset_valid, model, ema_model, optimizer, args
    )
