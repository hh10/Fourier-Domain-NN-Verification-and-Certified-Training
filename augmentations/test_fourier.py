import torch
from torchvision.utils import make_grid, save_image

import unittest
import numpy as np
import os
from itertools import product

from .fourier2d import fourier2d_augment
from .fourier2d_utils import polar2cart_bounds, unwrap_angle, wrap_angle, polar2z, get_freq_mask, dft2ztorchf
from .specification_networks import IDFT, IDFT_matrix


def ft_aug_fact(img, amp_noise, pha_noise, zero_centered=True):
    _, h, w = img.shape[-3:]
    img = img.numpy().transpose(1, 2, 0)

    img_fft = np.fft.fft2(img, axes=(0, 1)) / (h*w)
    fft_abs_noise = amp_noise.transpose(1, 2, 0)  # amplitude should be positive
    fft_pha_noise = pha_noise.transpose(1, 2, 0)
    
    if zero_centered:
        fft_noise = fft_abs_noise * (np.e ** (1j * wrap_angle(fft_pha_noise)))
        img_noise = np.real(np.fft.ifft2(fft_noise, axes=(0, 1)) * h*w)
        img_aug = img + img_noise
    else:
        img_abs, img_pha = np.abs(img_fft), unwrap_angle(np.angle(img_fft))
        img_abs = np.fft.fftshift(img_abs, axes=(0, 1))
        img_pha = np.fft.fftshift(img_pha, axes=(0, 1))
        img_abs += np.fft.fftshift(fft_abs_noise, axes=(0, 1))  # amplitude should be positive
        img_pha += np.fft.fftshift(fft_pha_noise, axes=(0, 1))
        img_abs = np.fft.ifftshift(img_abs, axes=(0, 1))
        img_pha = np.fft.ifftshift(img_pha, axes=(0, 1))
        img_aug = img_abs * (np.e ** (1j * wrap_angle(img_pha)))
        img_aug = np.real(np.fft.ifft2(img_aug, axes=(0, 1)) * h*w)

    img_aug = np.clip(img_aug, 0, 1)
    return torch.from_numpy(img_aug.transpose(2, 0, 1)).to(torch.float)


def fourier_single_augment(img, M, amp_noise, pha_noise, strategy="aug"):
    from .fourier import get_dft, polar2z
    amp, pha, z = get_dft(img)
    c, h, w = img.shape[-3:]
    if strategy == "aug":
        aug_amp, aug_pha = amp + amp_noise, pha + pha_noise
        return z, polar2z(aug_amp, aug_pha)
    amp_l, amp_u = np.maximum(amp - amp_noise, -1), np.minimum(amp + amp_noise, 1)
    pha_l, pha_u = pha - pha_noise, pha + pha_noise
    return z, polar2cart_bounds(amp_l, amp_u, pha_l, pha_u, assert_inside_z=z)


def DFT_matrix(N):
    # https://www.originlab.com/doc/Origin-Help/InverseFFT2-Algorithm
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(-2j * np.pi / N)
    W = np.power(omega, i * j)
    return W


class FourierAugTests(unittest.TestCase):
    test_dir = "/tmp/FDV_tests/fourier_augs"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.test_dir, exist_ok=True)

    def test_equivalance_with_FACT(self):
        n, R, BS = 29, 3, 2
        try:
            from PIL import Image
            img = Image.open("augmentations/datasets/253_0079.jpg").resize((n, n))
            img = (torch.from_numpy(np.array(img).transpose(2, 0, 1))/255.).unsqueeze(0)
            for _ in range(BS-1):
                img = torch.cat([img, img], dim=0)
        except:
            img = torch.zeros((BS, 3, n, n))
        for indiv_channels in [False, True]:
            _, aug, amp_noise, pha_noise = fourier2d_augment(img, freq_mask_config={'freq_mask_range': (0, R)}, pix_eps=0.313, indiv_channels=indiv_channels)
            print(amp_noise.max(), np.count_nonzero(amp_noise), pha_noise.max(), np.count_nonzero(pha_noise))

            img_aug_fact = torch.stack([ft_aug_fact(img[k], amp_noise[k], pha_noise[k]) for k in range(BS)])

            for concat_v in [True, False]:
                check_str = "" + ("_indiv_channels" if indiv_channels else "")
                print(f"Equiv:: IDFT version: concat_v: {concat_v}  R: {R} " + check_str)
                img_aug_own = IDFT(n, concat_v=concat_v, C=img.shape[-3])(aug, img)
                save_image(make_grid(torch.cat((img_aug_own, img_aug_fact))), f"{FourierAugTests.test_dir}/equivalence{check_str}.png")
                self.assertGreaterEqual(img_aug_own.min(), 0.)
                self.assertLessEqual(img_aug_own.max(), 1.)
                torch.testing.assert_close(img_aug_own, img_aug_fact)

    def test_ft_properties(self):
        n = 5
        img = torch.randint(10, (n, n))
        img_fft = np.fft.fft2(img, axes=(0, 1))
        img_real, img_imag = np.real(img_fft), np.imag(img_fft)
        img_abs = np.abs(img_fft)

        for mat in [img_real, img_abs]:
            mat = np.fft.fftshift(mat, axes=(0, 1))
            for i in range(mat.shape[0]//2):
                for j in range(mat.shape[1]//2):
                    np.testing.assert_almost_equal(mat[i, j], mat[mat.shape[0]-1-i, mat.shape[1]-1-j])
        mat = np.fft.fftshift(img_imag, axes=(0, 1))
        for i in range(mat.shape[0]//2):
            for j in range(mat.shape[1]//2):
                np.testing.assert_almost_equal(mat[i, j], -mat[mat.shape[0]-1-i, mat.shape[1]-1-j])
        z = np.concatenate((img_real, img_imag), axis=img.ndim-2)
        z_ = z[..., :z.shape[-1]//2+1]
        z_hat = IDFT.real_z(torch.from_numpy(z_))
        np.testing.assert_almost_equal(z_hat, z)

        # test for wrap_angle
        random_phi = (np.random.random((n, n)) - 0.5)*2*np.pi
        random_phi_hat = wrap_angle(random_phi)
        np.testing.assert_almost_equal(random_phi_hat, random_phi)

        # linearity in amp and phase property
        amp1 = np.random.random((3, n, n))
        amp2 = np.random.random((3, n, n))
        pha = (np.random.random((3, n, n))-0.5)*2 * np.pi

        amp_min, amp_max = np.minimum(amp1, amp2), np.maximum(amp1, amp2)
        real_min, real_max, imag_min, imag_max = polar2cart_bounds(amp_min, amp_max, pha, pha, cat_torchf=False)
        self.assertGreater(np.min(real_max-real_min), 0)
        self.assertGreater(np.min(imag_max-imag_min), 0)

        for amp in [amp1, amp2]:
            real, imag = polar2z(amp, pha, cat_torchf=False)
            self.assertGreaterEqual(np.min(real-real_min), 0)
            self.assertGreaterEqual(np.min(real_max-real), 0)
            self.assertGreaterEqual(np.min(imag-imag_min), 0)
            self.assertGreaterEqual(np.min(imag_max-imag), 0)

        amp_samples = [amp1 + np.random.random(amp1.shape)*(amp2-amp1) for _ in range(25)]
        for amp in amp_samples:
            real, imag = polar2z(amp, pha, cat_torchf=False)
            self.assertGreater(np.min(real-real_min), 0)
            self.assertGreater(np.min(real_max-real), 0)
            self.assertGreater(np.min(imag-imag_min), 0)
            self.assertGreater(np.min(imag_max-imag), 0)

        # idft_matrix properties
        IF = IDFT_matrix(n)
        torch.testing.assert_close(IF, IF.transpose())

        # dft
        z_hat_true = np.fft.fft2(img, axes=(img.ndim-2, img.ndim-1))
        F = DFT_matrix(n)
        z_hat_hat = F @ img.numpy() @ F.transpose()
        torch.testing.assert_close(z_hat_hat, z_hat_true)

        # idft
        x_hat_true = np.fft.ifft2(z_hat_true, axes=(img.ndim-2, img.ndim-1))
        torch.testing.assert_close(np.real(x_hat_true), img.double().numpy())
        z_hat_true_normalized = z_hat_true / np.prod(img.shape[-2:])
        x_hat_hat = IF @ z_hat_true_normalized @ IF.transpose()
        torch.testing.assert_close(x_hat_hat, x_hat_true)
        z_hat_true_normalized_cart = dft2ztorchf(z_hat_true_normalized).unsqueeze(0).unsqueeze(0)
        x_hat_linear1 = IDFT(n, concat_v=True, C=1).forward_concat(z_hat_true_normalized_cart)
        torch.testing.assert_close(np.clip(x_hat_linear1.detach().double().numpy()[0][0], 0, 1), np.clip(np.real(x_hat_true), 0, 1), rtol=1.3e-6, atol=1e-5)
        x_hat_linear2 = IDFT(n, concat_v=False, C=1).forward_non_concat_mm_lin(z_hat_true_normalized_cart)
        torch.testing.assert_close(np.clip(x_hat_linear2.detach().double().numpy()[0][0], 0, 1), np.clip(np.real(x_hat_true), 0, 1), rtol=1.3e-6, atol=1e-5)

        # multiplication
        shape = (n, n)
        z1 = np.sqrt(np.random.uniform(0, 1, shape)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, shape))
        amp1, pha1 = np.abs(z1), np.angle(z1)
        z2 = np.sqrt(np.random.uniform(0, 1, shape)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, shape))
        amp2, pha2 = np.abs(z2), np.angle(z2)
        z12 = z1 * z2
        np.testing.assert_almost_equal(np.abs(z12), amp1*amp2)
        np.testing.assert_almost_equal(unwrap_angle(np.angle(z12)), unwrap_angle(pha1+pha2))

    def test_fourier_bp(self):
        try:
            from .test_utils import compute_and_sanity_check_bounds_with_lirpa
            N, c, n, eps = 4, 1, 5, 0.05
            x = torch.rand((N, c, 2*n, n))*eps/2
            lb, ub = -torch.rand((N, c, 2*n, n))*eps, torch.rand((N, c, 2*n, n))*eps
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x, lb, ub = x.to(device), lb.to(device), ub.to(device)
            for concat_v, C in product([True, False], [1, 3]):
                print(f"BP:: IDFT version: concat_v {concat_v}")
                idft = IDFT(n, concat_v=concat_v, real_signal=False, C=C, device=device).to(device)
                c_x = (torch.ones((N, C, n, n)) * 0.5).to(device)
                x_hat = idft(x, c_x)
                # load it from verifiers -> compute bounds from at least 2 and compare
                # compute_bounds_with_verinet(idft, eps, input_shape, device)
                x_hat_lirpa, ilb_lirpa, iub_lirpa, _ = compute_and_sanity_check_bounds_with_lirpa(idft, (x, c_x), eps, lb, ub, device, test_dir=FourierAugTests.test_dir, IBP=True, method=None)
                # c = torch.eye(np.prod(x_hat.shape)).unsqueeze(0)
                # x_hat_lirpa, ilb_lirpa, iub_lirpa = compute_and_sanity_check_bounds_with_lirpa(idft, x, eps, lb, ub, device, IBP=False, method="forward+backward", C=c)
                save_image(make_grid(x_hat_lirpa), f"{FourierAugTests.test_dir}/fourier_bp_output.png")
                torch.testing.assert_close(x_hat_lirpa, x_hat)
                self.assertEqual(iub_lirpa.shape, x_hat.shape)
        except:
            print()
            print('#'*80)
            print("AutoLirpa not installed?")
            print('#'*80)

    def test_augmented_bp(self):
        try:
            from .test_utils import compute_and_sanity_check_bounds_with_lirpa
            from torch import nn
            import sys
            sys.path.insert(0, "..")
            from networks.network_impl import Normalization, MultiInputSequential

            n, eps = 5, 0.1
            input_shape = (3, 2*n, n)
            x, c_x = torch.rand((1, *input_shape))*eps/2, torch.ones((1, 3, n, n)) * 0.5
            lb = -torch.rand((1, *input_shape))*eps
            ub = torch.rand((1, *input_shape))*eps
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x, c_x, lb, ub = x.to(device), c_x.to(device), lb.to(device), ub.to(device)
            for concat_v in [True]:
                print(f"BP:: IDFT version: concat_v {concat_v}")
                idft = IDFT(n, concat_v=concat_v, real_signal=False, C=input_shape[-3], device=device).to(device)
                norm_net = Normalization(device, input_shape[0], mean=[0.5]*input_shape[0], std=[0.5]*input_shape[0])
                augmented_net = MultiInputSequential(idft, norm_net, nn.Conv2d(3, 3, 1, 1), nn.ReLU(), nn.Flatten(), nn.Linear(int(np.prod(input_shape)/2), 10)).to(device)
                # print(augmented_net)
                y_hat = augmented_net(x, c_x)

                c = torch.eye(np.prod(y_hat.shape)).unsqueeze(0).to(device)
                y_hat_lirpa, ilb_lirpa, iub_lirpa, _ = compute_and_sanity_check_bounds_with_lirpa(augmented_net, (x, c_x), eps, lb, ub, device, test_dir=FourierAugTests.test_dir, IBP=False, method="forward+backward", C=c)
                torch.testing.assert_close(y_hat_lirpa, y_hat)
        except:
            print()
            print('#'*80)
            print("AutoLirpa not installed?")
            print('#'*80)

    def test_freq_mask(self):
        shape = (3, 33, 33)
        Ms = []
        for R in [1, 2, 3]:
            M_ = get_freq_mask(shape, freq_mask_range=(0, R))
            M = np.fft.fftshift(M_, axes=(0, 1))
            M = torch.from_numpy(M).unsqueeze(0)
            print(f"# nonzero elements in M: {np.count_nonzero(M_)} and shifted_M: {np.count_nonzero(M)}")
            Ms.append(M)
        for r in [-1, -2, -3]:
            M = get_freq_mask(shape, freq_mask_range=(r, -1))
            M = np.fft.fftshift(M, axes=(0, 1))
            M = torch.from_numpy(M).unsqueeze(0)
            Ms.append(M)
        save_image(make_grid(Ms), f"{FourierAugTests.test_dir}/masks.png")


if __name__ == '__main__':
    unittest.main()
    # fat = FourierAugTests()
    # fat.setUpClass()
    # fat.test_equivalance_with_FACT()
    # fat.test_ft_properties()
    # fat.test_fourier_bp()
    # fat.test_augmented_bp()
    # fat.test_freq_mask()
