#  bound propagation widths through idft for different BP methods
import torch

import sys
import numpy as np

from .augmentations import InputAugmentor
from .test_utils import compute_and_sanity_check_bounds_with_lirpa
from data import get_dataset_info, get_dataset, get_data_loaders
from networks import get_network, AugmentedNetwork
from utils import load_model
from verifiers.auto_LiRPA.utils import get_spec_matrix


def stats(x, label=""):
    x = np.absolute(np.array((x)))
    return f"{label} min: {x.min()}, median: {np.median(x)}, mean: {x.mean()}, max: {x.max()}"


def rob_acc(y, y_hat, lb, ub, show_bounds=False):
    if show_bounds:
        for i in range(len(y)):
            print(f"Image {i} top-1 prediction {y[i]}")
            for j in range(lb.shape[-1]):
                print(f"f_{j}(x) = {y_hat[i][j]:8.3f},   {lb[i][j]:8.3f} <= f_{y[i]}(x_0+delta)-f_{j}(x_0+delta) <= {ub[i][j]:8.3f}")
            print()
    num_rob = len(y) - torch.sum((lb < 0).any(dim=1)).item()
    return num_rob


norm_ord = np.inf
pix_eps = 2/255  # 2/255, 8/255

models = {
    "nonrob": "1std/2023-12-25_18-37-42/best_model.pth",
    "pcert": "3pcert/2024-01-04_00-00-00/SABR_model.pth",
    "ft_cert_fb": "4ftcert_fb/2024-03-27_00-54-42/model_epoch150.pth",
    "ft_cert_ibp": "../configs_CIFAR10_ft_rob_ibp/2024-04-21_12-00-58/model_epoch364.pth",
}
model_path = f"results/ecai_paper_models/{models[sys.argv[1]]}"

dataset_name = "cifar10"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_info = get_dataset_info(name=dataset_name, image_size=33)
# idft = IDFT(data_info["shape"][-1], C=data_info["shape"][-3], real_signal=False)
# norm_net = Normalization("cpu", data_info["shape"][0], mean=[0.5]*data_info["shape"][0], std=[0.5]*data_info["shape"][0])
# network = MultiInputSequential(idft, nn.Flatten())  # , nn.Linear(int(np.prod(data_info["shape"])), 10))
model = get_network({"name": "CNN7", "bn": False, "bn2": False}, device, data_info)
load_model(model_path, device, model)

batch_size, indiv_channels = 16, False
test_datasets = get_dataset(False, data_info, dataset_name)
test_loaders = get_data_loaders("test", test_datasets, batch_size)
test_loader = list(test_loaders.values())[0]


def compute_post_idft_bounds(R):
    print("="*18 + f"\nR: {R}\n" + "="*18)
    aug_cfg = {
                "spec": "ball",
                "domain": "fourier",
                "pert_model": "additive",
                "strategy": "robust",
                "pix_eps": pix_eps,
                "norm_ord": norm_ord,
                "indiv_channels": indiv_channels,
                "fourier_mask": {"freq_mask_range": (0, R)},
            }

    Network = AugmentedNetwork(model, data_info, normalize=True, aug_cfg=aug_cfg, initialize=False)
    augmentor = InputAugmentor(data_info, aug_cfg)
    
    methods = ["FB"]  # , "FB"]
    widths, robs, seconds = {}, {}, {}
    for key in methods:
        widths[key], seconds[key], robs[key] = [], [], 0
    total = 0
    for bi, (img, y) in enumerate(test_loader):
        # if bi > 99:
        #     break

        # create appropriate augmented model
        network = Network.get_augmented_network(img)

        x, domain_bounds = augmentor.augment_and_process(None, img)[:2]
        x = x.to(device)
        total += len(y)

        # print("domain bounds:", torch.mean(domain_bounds[1]-domain_bounds[0]))
        z = torch.zeros((len(y), data_info["shape"][-3] if indiv_channels else 1, 2*data_info["shape"][-2], data_info["shape"][-1]))
        c = get_spec_matrix(z, y, 10)
        # c = torch.eye(10).unsqueeze(0).repeat(batch_size, 1, 1)
        # print(y), print(c)

        z, c, domain_bounds = z.to(device), c.to(device), domain_bounds.to(device)

        # x_hat_ba1, ilb1, iub1, time_s1 = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, domain_bounds[0], domain_bounds[1], device, IBP=True, method="ibp", C=c)
        # widths["IBP"] += [(iub1-ilb1).median().item()]
        # seconds["IBP"] += [time_s1]
        # robs["IBP"] += rob_acc(y, x_hat_ba1, ilb1, iub1)

        # x_hat_ba2, ilb2, iub2, time_s2 = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, domain_bounds[0], domain_bounds[1], device, method="CROWN-IBP", C=c)
        # # torch.testing.assert_close(x_hat_ba2, x_hat_ba1)
        # widths["CrIBP"] += [(iub2-ilb2).median().item()]
        # seconds["CrIBP"] += [time_s2]
        # robs["CrIBP"] += rob_acc(y, x_hat_ba2, ilb2, iub2)

        x_hat_ba3, ilb3, iub3, time_s3 = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, domain_bounds[0], domain_bounds[1], device, IBP=False, method="forward+backward", C=c)
        # torch.testing.assert_close(x_hat_ba3, x_hat_ba1)
        widths["FB"] += [(iub3-ilb3).median().item()]
        seconds["FB"] += [time_s3]
        robs["FB"] += rob_acc(y, x_hat_ba3, ilb3, iub3)

    for key in methods:
        print()
        print(stats(widths[key], key + " widths"))
        print(stats(seconds[key], key + " seconds"))
        print(f"accuracy: {robs[key]/total}")
        print()


if __name__ == "__main__":
    for R in [1, 2, 4]:  # , 2, 4, 8, 16]:
        compute_post_idft_bounds(R)
