from .template import *


def config(epochs, eps_linf, pert_model, R, **kwargs):
    spec_type = "ball"  # "segment"

    known_dataset = {
        "dataset_name": "cifar10",
        "batch_size": 16,
        "transform_strategy": "std",
        "normalize": dataset_normalization,
        "num_samples_per_input": 1,
    }
    datasets = {
        "info": {
            "name": "cifar10",
            "image_size": image_size,
        },
        "known_dataset": known_dataset,
        "target_dataset": target_dataset,
    }

    optimiser = {
        "type": "Adam",
        "lr": 5e-4,
        "scheduler": {
            "type": "MultiStepLR",
            "milestones": [105, 115],
            "gamma": 0.1
        },
    }
    training = {
        "seed": 0,
        "epoch": epochs,
        "optimiser": optimiser,
    }
    aug = {
        "spec": spec_type,
        "domain": "fourier",
        "pert_model": pert_model,
        "fourier_mask": {"freq_mask_range": (0, R)},
        "resolution_multiple": 1,
        "pix_eps": eps_linf,  # 0.14160052292963296, 0.5664020917185318, 1.1328041834370637
        "norm_ord": np.inf,
        "strategy": "robust",  # aug, adv, robust
        "indiv_channels": False,
        "pert_scheduling": {"profile": "sshaped", "end_epoch": 120, "start_epoch": 15, "beta": 1.5},
        "bounds_config": {"method": "forward+backward", "IBP": False},
        "minimum_robust_weight": 1,
    }

    return {
        "datasets": datasets,
        "networks": networks,
        "training": training,
        "aug": aug,
        "test_augs": {"Pixel8by255LinfAdv": test_adv, "Pixel8by255LinfAdvIllum": test_adv_illum}
    }
