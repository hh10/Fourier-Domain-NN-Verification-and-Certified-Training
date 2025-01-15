from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

import os

from .dataset_utils import GaussianNoise, CIFAR10C, AttributesDataset, MultiSampleDataset, CODaN


def get_transformer(strategy, shape, mean=None, std=None):
    transforms_list = []
    if strategy == "std":
        transforms_list += [
            transforms.RandomHorizontalFlip(),
            transforms.Resize(shape, antialias=True),
            transforms.RandomCrop(shape, padding=4, padding_mode='edge'),
            transforms.ToTensor(),
        ]
    elif strategy == "fair":
        transforms_list += [
            # transforms.Resize(int(image_size*1.1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.9, 1.15), contrast=(0.8, 1.2), saturation=(0.9, 1.1), hue=(-0.05, 0.05)),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.025, 0.05)),
            transforms.Resize(shape, antialias=True),
            transforms.RandomCrop(shape, padding=4, padding_mode='edge'),
            transforms.ToTensor(),
            GaussianNoise(std=0.005, p=0.5),
        ]
    elif strategy == "pixel_domain_shifting":
        transforms_list += [
            # transforms.Resize(int(image_size*1.1), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.75, 1.35), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)),
            transforms.RandomGrayscale(p=0.25),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.05, 2.0)),
            transforms.Resize(shape, antialias=True),
            transforms.RandomCrop(shape, padding=4, padding_mode='edge'),
            transforms.ToTensor(),
            GaussianNoise(std=0.01),
        ]
    elif strategy == "test":
        transforms_list += [transforms.Resize(shape, antialias=True), transforms.ToTensor()]
    else:
        raise NotImplementedError

    if mean is not None or std is not None:
        # normalise if mean and std provided
        transforms_list += [transforms.Normalize(mean or [0]*len(std), std or [1]*len(mean))]
    return transforms.Compose(transforms_list)


def get_dataset(train: bool, data_info, dataset_name, transform_strategy="test", subdomains=None, normalize=False, **kwargs):
    # dataset is returned in a dict
    # data is collated with attribute info added to the batch
    transform = get_transformer("test" if not train else transform_strategy, data_info["shape"][-2:], mean=data_info["mean"] if normalize else None, std=data_info["std"] if normalize else None)

    data_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../datasets", dataset_name))

    data = {}
    if dataset_name == "mnist":
        data[dataset_name] = datasets.MNIST(data_path, train=train, download=True, transform=transform)

    elif dataset_name == "tinyimagenet200":
        data[dataset_name] = datasets.ImageFolder(os.path.join(data_path, "train" if train else "val"), transform=transform)

    elif dataset_name == "trds":
        data[dataset_name] = datasets.ImageFolder(os.path.join(data_path, "train" if train else "val"), transform=transform)

    elif dataset_name == "cifar10" and (subdomains is None or "natural" in subdomains):
        data[dataset_name] = datasets.CIFAR10(data_path, train=train, download=True, transform=transform)

    elif dataset_name == "PACS":
        dataset_type = "train" if train else "val"
        domain_names = subdomains or os.listdir(os.path.join(data_path, dataset_type))
        domain_names.sort()

        # can also add functionality to make imbalanced datasets here
        datasets_class_to_idx = None
        for di, dn in enumerate(domain_names):
            dataset = datasets.ImageFolder(root=os.path.join(data_path, dataset_type, dn), transform=transform)
            datasets_class_to_idx = dataset.class_to_idx if datasets_class_to_idx is None else datasets_class_to_idx
            assert dataset.class_to_idx == datasets_class_to_idx, "Inconsistent class mappings in datasets"
            data[f"PACS_{dn}"] = AttributesDataset(dataset, {di: dn})

    elif dataset_name == "cifar10" and subdomains is not None:  # subdomains are corruption types in cifar
        for ci, cname in enumerate(subdomains):
            if cname == "natural":
                data[dataset_name] = AttributesDataset(data[dataset_name], (ci,))
                continue
            dataset = CIFAR10C(os.path.join(data_path, 'CIFAR-10-C'), cname, transform=transform, severity=kwargs.get("severity", None))
            data[f"cifar10_{cname}"] = AttributesDataset(dataset, {ci: cname})

    elif dataset_name == "codan":
        if not subdomains:
            data[dataset_name] = CODaN(data_path, split="train" if train else "val", transform=transform)
        else:
            for si, subdomain in enumerate(subdomains):
                dataset = CODaN(data_path, split=subdomain, transform=transform)
                data[f"{dataset_name}_{subdomain}"] = AttributesDataset(dataset, {si: subdomain})

    else:
        raise ValueError(f"dataset {dataset_name} not available")
    return data


def get_data_loaders(data_type: str, datasets: dict, batch_size, target_datasets: dict = None, use_cuda=True, num_samples_per_input=1, sample_from_datasets="all", **kwargs):
    assert not target_datasets or num_samples_per_input > 1, "why specify target_datasets if not requiring multiple samples per input"
    dl_cfg = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    if data_type in ["train", "val"]:
        # datasets are concatenated together (should only be the subdomains of the same dataset), and a single dataloader is returned
        if len(datasets) == 1:
            dataset, dataset_indices = list(datasets.values())[0], None
            if not target_datasets:
                assert sample_from_datasets in ["all", "intra"], "Cannot sample extra from different datasets as only one provided"

        else:
            datasets = datasets.values()
            dataset = ConcatDataset(datasets)  # check if single dataset is okay
            dataset_indices, last_ind = {}, 0
            for ds, ind in zip(datasets, dataset.cumulative_sizes):
                dataset_indices[ds.global_attr] = [last_ind, ind]
                last_ind = ind

        # make this work for a concat dataset
        if num_samples_per_input > 1:
            # extra samples from domains can be all, inter, intra or specific subdomain name for domain_transfer
            dataset = MultiSampleDataset(dataset, sample_from_datasets, subdataset_ranges=dataset_indices, sampling_datasets=target_datasets)
        dl_cfg["collate_fn"] = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(dataset, batch_size, shuffle=data_type == "train", drop_last=data_type == "train", **dl_cfg)

    else:
        dataloader = {}  # separate dataloaders are returned as a dict
        for dataset_name, dataset in datasets.items():
            if num_samples_per_input > 1:
                # extra samples from domains can be all, inter, intra or specific domain name for testing
                # note: really only use if using need to form segment queries for test data
                dataset = MultiSampleDataset(dataset, sample_from_datasets, sampling_datasets=target_datasets or datasets)
            dl_cfg["collate_fn"] = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
            dataloader[dataset_name] = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, **dl_cfg)
    return dataloader
