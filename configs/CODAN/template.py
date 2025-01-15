image_size = 33
num_classes = 10
dataset_normalization = False
dataset_name = "codan"
test_subdomains = ["test_night"]

target_dataset = {
    "dataset_name": dataset_name,
    "batch_size": 64,
    "normalize": dataset_normalization,
    "subdomains": test_subdomains,
    "num_samples_per_input": 1,
}
known_dataset_val = {
    "dataset_name": dataset_name,
    "batch_size": 1,  # because we will made segment queries
    "transform_strategy": "std",
    "normalize": dataset_normalization, 
    "num_samples_per_input": 2,  # used for test_augs evals
    "sample_from_datasets": "inter",  # valid only for segment
    "sample_datasets": test_subdomains,
}

networks = {
    "name": "CNN7",
    "bn": False,
    "bn2": False
}

segment_val_base = {
    "spec": "segment",
    "domain": "fourier",
    "pert_model": "additive",
    "fourier_mask": {"freq_mask_range": (0, 2)},
    "resolution_multiple": 1,
    "indiv_channels": False,
    "pix_eps": 0.5,
}

tests = {f"segment_{k}": {"strategy": k, **segment_val_base} for k in ["aug", "adv", "robust"]}
