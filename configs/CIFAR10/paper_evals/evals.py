from ..template import *
import os
import numpy as np


def config(eps_linf, cifar10c_severity, **kwargs):
    test_batch_size = 1

    target_dataset = {
        "dataset_name": "cifar10",
        "batch_size": test_batch_size,
        "normalize": dataset_normalization,
        "subdomains": None,  # ["brightness", "contrast"],  # [os.path.splitext(fn)[0] for fn in os.listdir("../../datasets/cifar10/CIFAR-10-C") if fn != "labels.npy"],
        "severity": int(cifar10c_severity),
    }
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
        "target_dataset": target_dataset,
    }

    domain = "fourier"  # "cosine", "fourier"
    norm_ord = np.inf
    indiv_channels = False

    def test_pix(pix_strategy):
        return {
            "spec": "ball",
            "domain": "pixel",
            "pert_model": "additive",
            "pix_eps": eps_linf,
            "norm_ord": norm_ord,
            "strategy": pix_strategy,
            "indiv_channels": indiv_channels,
        }

    def test_ft_masked(ft_strategy, R, **kwargs):
        return {
            "spec": "ball",
            "domain": domain,
            "pert_model": "additive",
            "fourier_mask": {"freq_mask_range": (0, R)},
            "resolution_multiple": 1,
            "pix_eps": eps_linf,
            "norm_ord": norm_ord,
            "strategy": ft_strategy,
            "indiv_channels": indiv_channels,
            # "bounds_config": {"method": "ibp", "IBP": True},
            **kwargs,
        }

    freq_Rs = [1, 2, 3, 4, 8, 16, None]
    ft_test_robs = {f"{domain}_rob_comp_{R}R": test_ft_masked("robust", R) for R in freq_Rs[:5]}
    ft_test_advs = {f"{domain}_adv_comp_{R}R": test_ft_masked("adv", R) for R in freq_Rs[:5]}
    ft_test_advs_LFBA = {f"{domain}_adv_LFBA_comp_{R}R": test_ft_masked("adv_LFBA", R) for R in freq_Rs}
    ft_test_augs = {f"{domain}_aug_comp_{R}R": test_ft_masked("aug", R) for R in freq_Rs}
    pix_tests = {"pix_rob": test_pix("robust"), "pix_adv": test_pix("adv"), "pix_aug": test_pix("aug")}

    return {
        "datasets": datasets,
        "test_augs": {
                      **ft_test_robs,
                      **ft_test_advs,
                      # **ft_test_advs_LFBA,
                      # **ft_test_augs,
                      **pix_tests
                    }
    }


if __name__ == "__main__":
    print(config(0.14160052292963296)["datasets"]["target_dataset"]["subdomains"])
