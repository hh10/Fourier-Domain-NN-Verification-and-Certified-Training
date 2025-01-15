import torch
import numpy as np

import os
import unittest
from tqdm import tqdm

from .specification_networks import IDFT
from .test_utils import compute_and_sanity_check_bounds_with_lirpa, compute_bounds_with_lirpa
from networks.network_impl import MultiInputSequential


def stats(x, label=""):
    return f"{label} min: {x.min().item()}, median: {x.median().item()}, mean: {x.mean().item()}, max: {x.max().item()}"


def get_idft_network(N=3, pert_model="additive"):
    from torch import nn
    idft = IDFT(N, C=1, real_signal=False, pert_model=pert_model)
    c = torch.eye(N*N).unsqueeze(0)
    network = MultiInputSequential(idft, nn.Flatten())  # , nn.Linear(N*N, 20), nn.ReLU(), nn.Linear(20, 10))
    return c, network


def get_idft_aug_network(N=3, pert_model="additive"):
    from torch import nn
    idft = IDFT(N, C=1, real_signal=False, pert_model=pert_model)
    network = MultiInputSequential(idft, nn.Flatten(), nn.Linear(N*N, 20), nn.ReLU(), nn.Linear(20, 10))
    c = torch.eye(10).unsqueeze(0)
    return c, network


def get_random_z_x_bounds(N, eps):
    z = torch.randn((1, 1, 2*N, N))
    x = torch.randn((1, 1, N, N))

    obounds = z.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    obounds[0, 0, 0, :N, :] = z[0, 0, :N, :] - eps/2
    obounds[1, 0, 0, :N, :] = z[0, 0, :N, :] + eps/2
    return z, x, obounds


class BoundsTests(unittest.TestCase):
    test_dir = "/tmp/FDV_tests/aug_bounds"
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 3
    x_rand = torch.rand((1, 1, N, N))
        
    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.test_dir, exist_ok=True)

    def test_brightness_bounds(self):
        N = BoundsTests.N
        c, network = get_idft_network(N)
        network = network.to(BoundsTests.device)

        xs = {"rand": BoundsTests.x_rand, "uniform": torch.ones((1, 1, N, N)) * 0.25}  # or anything uniform

        for xk, x in xs.items():
            z = torch.zeros((1, 1, 2*N, N))
            obounds = z.unsqueeze(0).repeat(2, 1, 1, 1, 1)
            for _ in range(25):
                # run test for multiple epsilons
                eps = np.random.uniform(1e-3, 4, 1)  # 4./255
                obounds[0, 0, 0, 0, 0] = z[0, 0, 0, 0] - eps/2
                obounds[1, 0, 0, 0, 0] = z[0, 0, 0, 0] + eps/2

                assert torch.max(obounds[1]-obounds[0]) > 0 and torch.min(obounds[1]-z) >= 0 and torch.min(z - obounds[0]) >= 0, f"{torch.min(obounds[1]-z)} {torch.min(z - obounds[0])}"

                print()
                print(f"brightness: x = {x}")
                x_hat_ba1, ilb1, iub1, _ = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, obounds[0], obounds[1], BoundsTests.device, test_dir=BoundsTests.test_dir, C=c, IBP=True, method="ibp")
                print("x1:", x_hat_ba1[0])
                print("ilb1:", ilb1[0])
                print("iub1:", iub1[0])
                print(stats(iub1-ilb1, "1:"))

                x_hat_ba3, ilb3, iub3, _ = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, obounds[0], obounds[1], BoundsTests.device, test_dir=BoundsTests.test_dir, C=c, IBP=False, method="forward+backward")
                print("x3:", x_hat_ba3[0])
                print("ilb3:", ilb3[0])
                print("iub3:", iub3[0])
                print(stats(iub3-ilb3, "3:"))

                if xk == "uniform":
                    # if uniform x, then brightness dft aug will change all bounds similarly
                    for lb in [ilb1, iub1, ilb3, iub3]:
                        for i in range(1, lb.shape[-1]):
                            torch.testing.assert_close(lb[0][i], lb[0][0])

    def test_contrast_bounds(self):
        N = BoundsTests.N
        c, network = get_idft_network(N, pert_model="spatial_mult")
        network = network.to(BoundsTests.device)

        xs = {"rand": BoundsTests.x_rand}  #, "uniform": torch.ones((1, 1, N, N)) * 0.25}  # or anything uniform

        for xk, x in xs.items():
            eps = np.random.uniform(1e-3, 2, 1)  # 4./255
            z = torch.zeros((1, 1, 2*N, N))
            obounds = z.unsqueeze(0).repeat(2, 1, 1, 1, 1)
            obounds[0, 0, 0, 0, 0] = z[0, 0, 0, 0] - eps/2
            obounds[1, 0, 0, 0, 0] = z[0, 0, 0, 0] + eps/2

            assert torch.max(obounds[1]-obounds[0]) > 0 and torch.min(obounds[1]-z) >= 0 and torch.min(z - obounds[0]) >= 0, f"{torch.min(obounds[1]-z)} {torch.min(z - obounds[0])}"

            print()
            print(f"contrast: x = {x}")
            x_hat_ba1, ilb1, iub1, _ = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, obounds[0], obounds[1], BoundsTests.device, test_dir=BoundsTests.test_dir, IBP=True, method="ibp")
            print("x1:", x_hat_ba1[0])
            print("ilb1:", ilb1[0])
            print("iub1:", iub1[0])
            print(stats(iub1-ilb1, "1:"))

            if xk == "uniform":
                # if uniform x, then brightness dft aug will change all bounds similarly
                for lb in [ilb1, iub1]:
                    for i in range(1, lb.shape[-1]):
                        torch.testing.assert_close(lb[0][i], lb[0][0])

    def test_bounds_soundness(self):
        N = 7
        for pert_model in ["additive", "spatial_mult"]:
            for network_fn in [get_idft_network, get_idft_aug_network]:
                print()
                print(f"Testing bounds' soundness for network {network_fn.__name__}")
                c, network = network_fn(N, pert_model=pert_model)
                device = BoundsTests.device
                network = network.to(device)

                for _ in tqdm(range(5)):
                    # run test for multiple epsilons
                    eps = np.random.uniform(1e-3, 2, 1)  # 8./255
                    for i in tqdm(range(50)):
                        z, x, obounds = get_random_z_x_bounds(N, eps)
                        # ibp
                        x_hat_ba1, ilb1, iub1, _ = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, obounds[0], obounds[1], device, test_dir=BoundsTests.test_dir, C=c, IBP=True, method="ibp")
                        if pert_model != "spatial_mult":
                            # forward+backward
                            x_hat_ba3, ilb3, iub3, _ = compute_and_sanity_check_bounds_with_lirpa(network, (z, x), None, obounds[0], obounds[1], device, C=c, IBP=False, method="forward+backward")

    def test_reinference_correctness(self):
        N, eps = 13, 8./255
        c, network = get_idft_aug_network(N)

        kwargs = {"device": BoundsTests.device, "IBP": True, "C": c, "method": "ibp"}
        # initialize a bounded network with different x
        z1, x1, obounds1 = get_random_z_x_bounds(N, eps)
        ba_network1, xhat1, lb1, ub1 = compute_and_sanity_check_bounds_with_lirpa(network, (z1, x1), None, obounds1[0], obounds1[1], test_dir=BoundsTests.test_dir, return_bounded_model=True, **kwargs)

        # reinitialize another bounded network with different x
        z2, x2, obounds2 = get_random_z_x_bounds(N, eps)
        ba_network2, xhat2, lb2, ub2 = compute_and_sanity_check_bounds_with_lirpa(network, (z2, x2), None, obounds2[0], obounds2[1], return_bounded_model=True, **kwargs)

        # pass different x from first model and see if it okay
        xhat1_2, lb1_2, ub1_2, _ = compute_bounds_with_lirpa(ba_network2, (z1, x1), None, obounds1[0], obounds1[1], **kwargs)
        xhat2_1, lb2_1, ub2_1, _ = compute_bounds_with_lirpa(ba_network1, (z2, x2), None, obounds2[0], obounds2[1], **kwargs)

        torch.testing.assert_close(xhat1_2, xhat1)
        torch.testing.assert_close(lb1_2, lb1)
        torch.testing.assert_close(ub1_2, ub1)

        torch.testing.assert_close(xhat2_1, xhat2)
        torch.testing.assert_close(lb2_1, lb2)
        torch.testing.assert_close(ub2_1, ub2)


if __name__ == '__main__':
    unittest.main()
    # bt = BoundsTests()
    # bt.setUpClass()
    # bt.test_contrast_bounds()
    # bt.test_brightness_bounds()
    # bt.test_bounds_soundness()
    # bt.test_reinference_correctness()
