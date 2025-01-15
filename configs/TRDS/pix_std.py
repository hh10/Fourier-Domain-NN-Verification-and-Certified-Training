from .template import *


def config(epochs, **kwargs):
    known_dataset = {
        "dataset_name": "trds",
        "batch_size": 32,
        "transform_strategy": "fair",
        "normalize": dataset_normalization,  # normalized in the network
        "num_samples_per_input": 1,
    }
    datasets = {
        "info": {
            "name": "trds",
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
            "milestones": [int(ms*epochs) for ms in [0.75, 0.9]],
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
        "aug": {},
        "test_augs": {"Pixel8by255L2Adv": test_adv, "Pixel8by255L2AdvIllum": test_adv_illum}
    }
