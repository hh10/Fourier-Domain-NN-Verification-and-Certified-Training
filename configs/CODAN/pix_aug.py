from .template import *


def config(epochs, eps_l2):
    known_dataset_train = {
        "dataset_name": dataset_name,
        "batch_size": 64,
        "transform_strategy": "std",
        "normalize": dataset_normalization,
        "num_samples_per_input": 2,
        "sample_from_datasets": "inter",  # valid only for segment
        "sample_datasets": test_subdomains,
    }
    datasets = {
        "info": {
            "name": dataset_name,
            "image_size": image_size,
        },
        "known_dataset": {"train": known_dataset_train, "val": known_dataset_val},
        "target_dataset": target_dataset,
    }

    optimiser = {
        "type": "Adam",
        "lr": 5e-4,
        "scheduler": {
            "type": "MultiStepLR",
            "milestones": [int(ms*epochs) for ms in [0.75, 0.9]],
            "gamma": 0.1
        },
    }
    training = {
        "seed": 0,
        "epoch": epochs,
        "optimiser": optimiser,
    }

    aug = {
        "spec": "segment",
        "domain": "fourier",
        "pert_model": "additive",
        "pix_eps": eps_l2,
        "strategy": "aug",
        "fourier_mask": {"freq_mask_range": (0, 4)},
        "indiv_channels": False,
        "pert_scheduling": {"profile": "sshaped", "end_epoch": int(0.9*epochs), "start_epoch": int(0.025*epochs)}
    }

    return {
        "datasets": datasets,
        "networks": networks,
        "training": training,
        "aug": aug,
        "test_augs": tests,
    }
