import numpy as np

image_size = 33
spec_type = "ball"  # "segment"
num_classes = 10
dataset_normalization = False

target_dataset = {
    "dataset_name": "cifar10",
    "batch_size": 64,
    "normalize": dataset_normalization,
    "subdomains": None,  # ["brightness", "contrast"],
    "severity": 5,
}
known_dataset = {
    "dataset_name": "cifar10",
    "batch_size": 64,
    "transform_strategy": "std",
    "normalize": dataset_normalization, 
    "num_samples_per_input": 2 if spec_type == "segment" else 1,
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

bn = False
# networks = {
#     "name": "resnet10",
#     "linear_layer_in": 256,
#     "classes": 10
# }
# networks = {
#     "name": "CNN7",
#     "bn": bn,
#     "bn2": bn
# }
networks = {
    "name": "cifar_deep"
}

test_adv = {
    "spec": spec_type,
    "domain": "pixel",
    "pert_model": "additive",
    "pix_eps": 8/255,
    "norm_ord": np.inf,
    "strategy": "adv",
    "indiv_channels": True,
}
test_adv_illum = {
    "spec": spec_type,
    "domain": "pixel",
    "pert_model": "additive",
    "pix_eps": 8/255,
    "norm_ord": np.inf,
    "strategy": "adv",
    "indiv_channels": False,
}

# optimiser = {
#     "type": "SGD",
#     "lr": 1e-3,
#     "momentum": 0.9,
#     "weight_decay": 5e-4,
#     "nesterov": True,
#     "scheduler": {
#         "type": "StepLR",
#         "step_size": int(epoch * 0.85),
#         "gamma": 0.1
#     },
# }
# optimiser = {
#     "type": "Adam",
#     "lr": 5e-4,
#     "scheduler": {
#         "type": "MultiStepLR",
#         "milestones": [int(ms*epoch) for ms in [0.7, 0.85]],
#         "gamma": 0.1
#     },
# }
