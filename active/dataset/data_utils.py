from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from .augment import RandAugment

CACHE_PATH = "./data/"

NORM_MEAN = {
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
    "svhn": [0.4380, 0.4440, 0.4730],
}
NORM_STD = {
    "cifar10": [0.2470, 0.2435, 0.2616],
    "cifar100": [0.2673, 0.2564, 0.2762],
    "svhn": [0.1751, 0.1771, 0.1744],
}


def get_transform(export_id, is_train, use_strong):
    tf_lst = []

    if is_train:
        tf_lst += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if use_strong:
            tf_lst += [RandAugment(n=3)]

    mean, std = NORM_MEAN[export_id], NORM_STD[export_id]

    tf_lst += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(tf_lst)


class MyDataset:
    def __init__(self, export_id, is_train, use_single=True, use_strong=False):
        root = f"{CACHE_PATH + export_id}"

        if "cifar" in export_id:
            self.data = globals()[export_id.upper()](root, is_train, download=True)
            self.targets = self.data.targets
        elif "svhn" in export_id:
            split = "train" if is_train else "test"
            self.data = globals()[export_id.upper()](root, split, download=True)
            self.targets = self.data.labels.tolist()
        else:
            raise NotImplementedError

        self.tf_weak = get_transform(export_id, is_train, use_strong=False)
        if use_strong:
            self.tf_strong = get_transform(export_id, is_train, use_strong=True)

        self.use_single = use_single
        self.use_strong = use_strong

    def __getitem__(self, index):
        sample, target = self.data.__getitem__(index)
        image = self.tf_weak(sample)

        if self.use_single:
            ra_image = -1
        else:
            if self.use_strong:
                ra_image = self.tf_strong(sample)
            else:
                ra_image = self.tf_weak(sample)

        return image, ra_image, target, index

    def __len__(self):
        return len(self.targets)


class MySubset(Subset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.targets = [self.dataset.targets[idx] for idx in self.indices]

    def __getitem__(self, index):
        image, ra_image, target, _ = super().__getitem__(index)
        return image, ra_image, target, index
