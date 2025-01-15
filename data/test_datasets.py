import torch
from torchvision.utils import make_grid, save_image

import os
import unittest

from .dataset_utils import get_dataset_info
from .datasets import get_dataset, get_data_loaders


def run_test_config(dataset_name, test_subdomains):
    dataset_cfg = {
        "info": {
            "name": dataset_name,
            "image_size": 33,
        },
        "target_dataset": {
            "dataset_name": dataset_name,
            "subdomains": test_subdomains,
            "batch_size": 4,
            "transform_strategy": "std",
            "num_samples_per_input": 2,
            "sample_from_datasets": "inter",
        },
    }
    print("test_subdomains", test_subdomains)
    data_info = get_dataset_info(**dataset_cfg["info"])  # todo(hh): also send in normalization data
    dataset_test = get_dataset(False, data_info, **dataset_cfg["target_dataset"])
    tl_loader = get_data_loaders("test", dataset_test, **dataset_cfg["target_dataset"])
    assert len(dataset_test) == len(test_subdomains) and len(tl_loader) == len(test_subdomains), f"{len(dataset_test)} {len(tl_loader)}"
    batch = next(iter(tl_loader[f"{dataset_name}_{test_subdomains[0]}"]))
    x, y, a, x1, y1, a1 = batch
    print(y[0], a[0], y1[0], a1[0])
    print("test:", x.shape, y.shape, x1.shape, y1.shape)
    return dataset_test


def run_single_training_config(dataset_name):
    dataset_cfg = {
        "info": {
            "name": dataset_name,
            "image_size": 33,
        },
        "known_dataset": {
            "dataset_name": dataset_name,
            "batch_size": 4,
            "transform_strategy": "std",
            "num_samples_per_input": 1,
        },
    }
    data_info = get_dataset_info(**dataset_cfg["info"])  # todo(hh): also send in normalization data

    dataset_train = get_dataset(True, data_info, **dataset_cfg["known_dataset"])
    assert len(dataset_train) == 1
    tl_loader = get_data_loaders("train", dataset_train, **dataset_cfg["known_dataset"])
    batch = next(iter(tl_loader))
    x, y = batch
    print("single train:", x.shape, y.shape)
    # assert all([a1_train[i] != ai for i, ai in enumerate(a_train)]), f"{a_train} {a1_train}"

    dataset_val = get_dataset(False, data_info, **dataset_cfg["known_dataset"])
    tl_loader = get_data_loaders("val", dataset_val, **dataset_cfg["known_dataset"])
    batch = next(iter(tl_loader))
    x, y = batch
    print("single val:", x.shape, y.shape)
    # assert any([a1_val[i] != ai for i, ai in enumerate(a_val)]) and any([a1_val[i] == ai for i, ai in enumerate(a_val)]), f"{a_val} {a1_val}"
    assert len(list(dataset_val.values())[0]) < len(list(dataset_train.values())[0])


def run_conditional_training_config(dataset_name, target_datasets):
    dataset_cfg = {
        "info": {
            "name": dataset_name,
            "image_size": 33,
        },
        "known_dataset": {
            "dataset_name": "codan",
            "batch_size": 4,
            "transform_strategy": "std",
            "num_samples_per_input": 2,
            "sample_datasets": list(target_datasets.keys()),
        },
    }
    data_info = get_dataset_info(**dataset_cfg["info"])  # todo(hh): also send in normalization data

    dataset_train = get_dataset(True, data_info, **dataset_cfg["known_dataset"])
    assert len(dataset_train) == 1
    if "sample_datasets" in dataset_cfg["known_dataset"]:
        sample_datasets_list, dataset_cfg["known_dataset"]["target_datasets"] = dataset_cfg["known_dataset"].pop("sample_datasets"), {}
        for k in sample_datasets_list:
            dataset_cfg["known_dataset"]["target_datasets"][k] = target_datasets[k]
    train_loader = get_data_loaders("train", dataset_train, **dataset_cfg["known_dataset"])
    batch = next(iter(train_loader))
    print(len(batch))
    x, y, a, x1, y1, a1 = batch
    print(y[0], a[0], y1[0], a1[0])
    print("conditional train:", x.shape, y.shape, x1.shape, y1.shape)
    save_image(make_grid(torch.cat([x, x1])), f"{DatasetTests.test_dir}/{dataset_name}_conditional_train.png")

    dataset_val = get_dataset(False, data_info, **dataset_cfg["known_dataset"])
    val_loader = get_data_loaders("val", dataset_val, **dataset_cfg["known_dataset"])
    batch = next(iter(val_loader))
    x, y, a, x1, y1, a1 = batch
    print(y[0], a[0], y1[0], a1[0])
    print("conditional val:", x.shape, y.shape, x1.shape, y1.shape)
    save_image(make_grid(torch.cat([x, x1])), f"{DatasetTests.test_dir}/{dataset_name}_conditional_val.png")

    # assert any([a1_val[i] != ai for i, ai in enumerate(a_val)]) and any([a1_val[i] == ai for i, ai in enumerate(a_val)]), f"{a_val} {a1_val}"
    assert len(list(dataset_val.values())[0]) < len(list(dataset_train.values())[0])
    return train_loader, val_loader


class DatasetTests(unittest.TestCase):
    test_dir = "/tmp/FDV_tests/datasets"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.test_dir, exist_ok=True)

    def test_codan(self):
        datasets_to_test = {
            "cifar10": ["fog", "impulse_noise"],
            "codan": ["test_night"],
        }
        for d2t_name, d2t_test_subdomains in datasets_to_test.items():
            dataset_test = run_test_config(d2t_name, d2t_test_subdomains)
            run_single_training_config(d2t_name)
            run_conditional_training_config(d2t_name, dataset_test)


if __name__ == "__main__":
    unittest.main()
