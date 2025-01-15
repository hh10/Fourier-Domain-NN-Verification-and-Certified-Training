from ..template import *
import os
import numpy as np


def config(eps_l2, **kwargs):
    test_batch_size = 8

    known_dataset = {
        "dataset_name": "cifar10",
        "batch_size": test_batch_size,
        "transform_strategy": "std",
        "normalize": dataset_normalization,
        "num_samples_per_input": 1,
        "sample_from_datasets": "all",  # valid only for segment
    }
    datasets = {
        "info": {
            "name": "cifar10",
            "image_size": image_size,
        },
        "known_dataset": known_dataset,
        "target_dataset": known_dataset,
    }

    norm_ord = 2

    def test_ft_masked(ft_strategy, R):
        return {
            "spec": "segment",
            "domain": "fourier",
            "pert_model": "additive",
            "fourier_mask": {"freq_mask_range": (0, R)},
            "resolution_multiple": 1,
            "pix_eps": eps_l2,
            "norm_ord": norm_ord,
            "strategy": ft_strategy,
            "indiv_channels": True,
            "zero_centered": False,
            # "bounds_config": {"method": "ibp", "IBP": True},
        }

    # fourier_Rs = [1,],  # , 2, 3, 4, 8, 16, None]  # , 8]  # , -1]
    ft_test_augs = {f"fourier_rob_comp_{R}R": test_ft_masked("robust", R) for R in [1,]}

    return {
        "datasets": datasets,
        "test_augs": {
                      **ft_test_augs,
                    }
    }


if __name__ == "__main__":
    print(config(0.14160052292963296)["datasets"]["target_dataset"]["subdomains"])
