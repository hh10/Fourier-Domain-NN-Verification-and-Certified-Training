from .template import *


def config(epochs, eps_l2, pert_model, R):
    spec_type = "ball"  # "segment"

    known_dataset = {
        "dataset_name": "cifar10",
        "batch_size": 256,
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
            "milestones": [85, 110],
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
        "pix_eps": eps_l2,  # 0.14160052292963296, 0.5664020917185318, 1.1328041834370637
        "norm_ord": 2,
        "strategy": "aug",  # aug, adv, robust
        "indiv_channels": False,
        "zero_centered": True,
        "pert_scheduling": {"profile": "sshaped", "end_epoch": 115, "start_epoch": 9, "beta": 1.5}
    }

    return {
        "datasets": datasets,
        "networks": networks,
        "training": training,
        "aug": aug,
        "test_augs": {"Pixel8by255L2Adv": test_adv, "Pixel8by255L2AdvIllum": test_adv_illum}
    }
