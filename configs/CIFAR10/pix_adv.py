from .template import *


def config(epochs, eps_linf, **kwargs):
    spec_type = "ball"  # "segment"

    known_dataset = {
        "dataset_name": "cifar10",
        "batch_size": 256,
        "transform_strategy": "std",
        "normalize": dataset_normalization,  # normalized in the network
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
        "weight_decay": 5e-5,
        "scheduler": {
            "type": "MultiStepLR",
            "milestones": [110, 120],
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
        "domain": "pixel",
        "pert_model": "additive",
        "pix_eps": eps_linf,  # 0.14160052292963296, 0.5664020917185318
        "norm_ord": np.inf,
        "strategy": "adv",  # aug, adv, robust
        "indiv_channels": False,
        "pert_scheduling": {"profile": "sshaped", "end_epoch": 120, "start_epoch": 11, "beta": 1.5}
    }

    return {
        "datasets": datasets,
        "networks": networks,
        "training": training,
        "aug": aug,
        "test_augs": {"Pixel8by255LinfAdv": test_adv, "Pixel8by255LinfAdvIllum": test_adv_illum}
    }
