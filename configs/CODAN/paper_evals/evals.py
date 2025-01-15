from ..template import *


def config(**kwargs):
    datasets = {
        "info": {
            "name": dataset_name,
            "image_size": image_size,
        },
        "known_dataset": {"val": known_dataset_val},
        "target_dataset": target_dataset,
    }

    def test_ft_masked(pix_eps, ft_strategy, R):
        return {
            "spec": "segment",
            "domain": "fourier",
            "pert_model": "additive",
            "fourier_mask": {"freq_mask_range": (0, R)},
            "resolution_multiple": 1,
            "pix_eps": pix_eps,
            "strategy": ft_strategy,
            "indiv_channels": ft_strategy == "adv",
        }

    pix_epses = [0.01, 0.05, 0.1]
    fourier_Rs = [1, 2, 3]  # , 8]  # , -1]
    ft_test_robs = {f"fourier_rob_comp_{R}R_eps{pix_eps}": test_ft_masked(pix_eps, "robust", R) for R in fourier_Rs for pix_eps in pix_epses}
    ft_test_advs = {f"fourier_adv_comp_{R}R_eps{pix_eps}": test_ft_masked(pix_eps, "adv", R) for R in fourier_Rs for pix_eps in pix_epses}
    ft_test_augs = {f"fourier_aug_comp_{R}R_eps{pix_eps}": test_ft_masked(pix_eps, "aug", R) for R in fourier_Rs for pix_eps in pix_epses}
    # pix_tests = {"pix_rob": test_pix("robust"), "pix_adv": test_pix("adv"), "pix_aug": test_pix("aug")}

    return {
        "datasets": datasets,
        "test_augs": {**ft_test_robs, **ft_test_advs, **ft_test_augs}
        # "test_augs": {"pix_rob": test_pix("robust"), "fourier_rob_comp_2R": test_ft_masked("robust", 2)}
    }


if __name__ == "__main__":
    print(config["datasets"]["target_dataset"]["subdomains"])
