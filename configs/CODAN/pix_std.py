from .template import *


def config(epochs, **kwargs):
    known_dataset_train = {
        "dataset_name": dataset_name,
        "batch_size": 128,
        "transform_strategy": "fair",
        "normalize": dataset_normalization,
        "num_samples_per_input": 1,
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
            "milestones": [int(ms*epochs) for ms in [0.7, 0.85]],
            "gamma": 0.1
        },
    }
    training = {
        "seed": 0,
        "epoch": epochs,
        "optimiser": optimiser,
    }

    return {
        "datasets": datasets,
        "networks": networks,
        "training": training,
        "aug": None,
        "test_augs": tests,
    }
