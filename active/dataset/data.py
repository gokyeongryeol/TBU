import numpy as np
import torch

from .data_utils import MyDataset, MySubset


def get_dataset_train(data, args):
    dataset_lab = data.get_labeled_data()
    dataset_unlab = data.get_unlabeled_data(use_ws=True)[1] if args.semi else None

    return dataset_lab, dataset_unlab


def get_dataset_valid(data):
    dataset_valid = data.get_valid_data()

    return dataset_valid


def prepare_data(rd, args):
    data = DataPool(args.export_id)
    data.initialize_labels(args.n_init, args.seed)

    for rd_ in range(1, rd):
        idxs_path = f"./saved/idxs/{args.path}/round_{rd_}.pt"
        data.update(torch.load(idxs_path))

    return data


class DataPool:
    def __init__(self, export_id):
        self.train_data = MyDataset(
            export_id, is_train=True, use_single=True, use_strong=False
        )
        self.valid_data = MyDataset(
            export_id, is_train=False, use_single=True, use_strong=False
        )
        self.ws_pair_data = MyDataset(
            export_id, is_train=True, use_single=False, use_strong=True
        )

        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_data)

        self.num_classes = len(np.unique(self.train_data.targets))
        self.labeled_idxs = None

    def initialize_labels(self, n_init, seed):
        self.labeled_idxs = np.zeros(self.n_train, dtype=bool)

        init_idxs = np.arange(self.n_train)
        np.random.seed(seed)
        np.random.shuffle(init_idxs)
        self.update(init_idxs[:n_init])

    def update(self, q_idxs):
        self.labeled_idxs[q_idxs] = True

    def get_indexed_data(self, dataset, indices):
        return MySubset(dataset, indices)

    def get_labeled_data(self):
        idxs_lab = np.arange(self.n_train)[self.labeled_idxs]
        return self.get_indexed_data(self.train_data, idxs_lab)

    def get_unlabeled_data(self, use_ws=False):
        idxs_unlab = np.arange(self.n_train)[~self.labeled_idxs]
        if use_ws:
            train_data = self.ws_pair_data
        else:
            train_data = self.train_data
        return idxs_unlab, self.get_indexed_data(train_data, idxs_unlab)

    def get_valid_data(self):
        return self.valid_data
