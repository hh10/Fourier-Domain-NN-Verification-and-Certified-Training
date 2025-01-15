import torch
from torchvision.utils import make_grid, save_image

import unittest
import os
import numpy as np
from scipy.fft import dct, idct, dctn, idctn

from .cosine2d import dct2d_augment
from .specification_networks import IDCT


class DCTAugTests(unittest.TestCase):
    test_dir = "/tmp/FDV_tests/dct_augs"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.test_dir, exist_ok=True)

    def test_dct_properties(self):
        N, norm = 5, 'ortho'
        x = (np.random.rand(N, N)-0.5)*2
        x_dct2 = dctn(x, norm=norm)
        x_dct2_hat = dct(dct(x.T, norm=norm).T, norm=norm)
        torch.testing.assert_close(x_dct2_hat, x_dct2)
        x_hat = idctn(x_dct2, norm=norm)
        torch.testing.assert_close(x_hat, x)
        x_hat2 = idct(idct(x_dct2.T, norm=norm).T, norm=norm)
        torch.testing.assert_close(x_hat2, x)

        D = dct(np.eye(N), axis=0, norm=norm)
        Dinv = idct(np.eye(N), axis=0, norm=norm)
        torch.testing.assert_close(Dinv, D.T)

        x_dct2mat_hat = D @ (x @ D.T)
        torch.testing.assert_close(x_dct2mat_hat, x_dct2)
        x_hat2mat = Dinv @ (x_dct2 @ Dinv.T)
        torch.testing.assert_close(x_hat2mat, x)
        x_hat3mat = ((x_dct2 @ Dinv.T).T @ Dinv.T).T
        torch.testing.assert_close(x_hat3mat, x)

    def test_equivalance_with_scipy(self):
        n, R, BS = 33, 2, 2
        try:
            from PIL import Image
            img = Image.open("augmentations/datasets/253_0079.jpg").resize((n, n))
            img = (torch.from_numpy(np.array(img).transpose(2, 0, 1))/255.).unsqueeze(0)
            for _ in range(BS-1):
                img = torch.cat([img, img], dim=0)
        except:
            img = torch.zeros((BS, 3, n, n))
        for indiv_channels in [False, True]:
            dct2, dct2_aug = dct2d_augment(img, freq_mask_config={'freq_mask_range': (0, R)}, eps=0.75, indiv_channels=indiv_channels)
            
            idct_own = IDCT(n, C=img.shape[-3], clipping_net_type=None)
            img_hat = idct_own(dct2)
            torch.testing.assert_close(img_hat, img)
            
            img_aug_scipy = torch.FloatTensor(idctn(dct2_aug.numpy(), axes=[2, 3], norm='ortho'))

            check_str = "" + ("_indiv_channels" if indiv_channels else "")
            print(f"Equiv:: IDCT R: {R} " + check_str)
            img_aug_own = idct_own(dct2_aug)
            save_image(make_grid(torch.cat((img_aug_own, img_aug_scipy))), f"{DCTAugTests.test_dir}/equivalence{check_str}.png")
            torch.testing.assert_close(img_aug_own, img_aug_scipy)

    def test_cosine_bp(self):
        try:
            from .test_utils import compute_and_sanity_check_bounds_with_lirpa
            N, c, n, eps = 4, 1, 5, 0.05
            x = torch.rand((N, c, n, n))*eps/2
            lb, ub = -torch.rand((N, c, n, n))*eps, torch.rand((N, c, n, n))*eps
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x, lb, ub = x.to(device), lb.to(device), ub.to(device)
            for C in [1, 3]:
                idct_net = IDCT(n, C=C).to(device)
                x_hat = idct_net(x)
                # load it from verifiers -> compute bounds from at least 2 and compare
                # compute_bounds_with_verinet(idft, eps, input_shape, device)
                x_hat_lirpa, ilb_lirpa, iub_lirpa, _ = compute_and_sanity_check_bounds_with_lirpa(idct_net, x, eps, lb, ub, device, test_dir=DCTAugTests.test_dir, IBP=True, method=None)
                # c = torch.eye(np.prod(x_hat.shape)).unsqueeze(0)
                # x_hat_lirpa, ilb_lirpa, iub_lirpa = compute_and_sanity_check_bounds_with_lirpa(idct_net, x, eps, lb, ub, device, IBP=False, method="forward+backward", C=c)
                save_image(make_grid(x_hat_lirpa), f"{DCTAugTests.test_dir}/fourier_bp_output.png")
                torch.testing.assert_close(x_hat_lirpa, x_hat)
                self.assertEqual(iub_lirpa.shape, x_hat.shape)
        except:
            print()
            print('#'*80)
            print("AutoLirpa not installed?")
            print('#'*80)

    def test_augmented_cosine_bp(self):
        try:
            from .test_utils import compute_and_sanity_check_bounds_with_lirpa
            from torch import nn
            import sys
            sys.path.insert(0, "..")
            from networks.network_impl import Normalization

            n, eps = 5, 0.1
            input_shape = (3, n, n)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            x = (torch.rand((1, *input_shape))*eps/2).to(device)
            idct_net = IDCT(n, C=input_shape[-3]).to(device)
            y_hat = idct_net(x)

            lb = -torch.rand((1, *input_shape))*eps
            ub = torch.rand((1, *input_shape))*eps
            lb, ub = lb.to(device), ub.to(device)
            norm_net = Normalization(device, input_shape[0], mean=[0.5]*input_shape[0], std=[0.5]*input_shape[0])
            augmented_net = nn.Sequential(idct_net, norm_net, nn.Conv2d(3, 3, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(np.prod(input_shape), 10)).to(device)
            # print(augmented_net)
            y_hat = augmented_net(x)

            c = torch.eye(np.prod(y_hat.shape)).unsqueeze(0).to(device)
            # y_hat_lirpa, ilb_lirpa, iub_lirpa = compute_and_sanity_check_bounds_with_lirpa(augmented_net, x, eps, lb, ub, device, test_dir=DCTAugTests.test_dir, IBP=True, method="ibp", C=c)
            y_hat_lirpa, ilb_lirpa, iub_lirpa, _ = compute_and_sanity_check_bounds_with_lirpa(augmented_net, x, eps, lb, ub, device, test_dir=DCTAugTests.test_dir, IBP=False, method="forward+backward", C=c)
            torch.testing.assert_close(y_hat_lirpa, y_hat)
        except:
            print()
            print('#'*80)
            print("AutoLirpa not installed?")
            print('#'*80)


if __name__ == "__main__":
    unittest.main()
