import torch
import numpy as np

from .attacks import fourier_pgd_zerocentered_amp_attack, zerocentered_pert_reproject, pgd_attack, LFBA_zero_centered_attack
from .fourier2d_utils import (
    get_freq_mask,
    eps_pixinf2ftinf,
    get_dft,
    polar2z,
    polar2cart_zerocentered_amp_bounds,
    noise,
    dft2ztorchf,
    polar2cart_bounds,
    get_kernel_range,
    mult_bound,
)


def fourier2d_augment(img1, pert_model="additive", pix_eps=None, M=None, freq_mask_config=None, kernel_dft_range=None, kernel_config=None, **kwargs):
    # fourier augmentation strategies:
    # 1. for illumination (comparison to bias-fields), do NO_indiv_channels, central freq radius (5), amplitude_eps (0.1) & phase_eps (-pi to pi) --- for RT
    # 2. for domain (*) (comparison to empirical work and robust training in pixel space), do INDIV_channels
    #   a. (*generalization) central freq radius (5), amplitude_eps (0.1) & phase_eps (-pi to pi) --- for RT
    #   b. (*transfer) central freq radius (5), amplitude_eps as governed by other domain amplitudes (no phase_eps) --- for RT
    # for verification:
    # 1. speaks of all possible illumination changes with allowed frequency characs!
    # 2.b  speaks of certain amplitude changes from original image to a different domain (+plus if the phase is changed for even 5, it is significant)
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if pert_model == "spatial_conv":
        # no resolution change allowed
        return fourier2d_kernel_augment(img1, kernel_dft_range=kernel_dft_range, kernel_config=kernel_config, **kwargs)
    return fourier2d_addmult_augment(img1, pix_eps, M=M, freq_mask_config=freq_mask_config, pert_model=pert_model, **kwargs)


def fourier2d_addmult_augment(img1, pix_eps, M=None, freq_mask_config=None, img2=None, resolution_multiple=1, pert_model="additive", norm_ord=np.inf, **kwargs):
    c, h, w = img1.shape[-3:]
    if M is None:
        M = get_freq_mask((c, h, w), **freq_mask_config, resolution_multiple=resolution_multiple)

    if img2 is not None:
        return fourier2d_pair_augment(img1, img2, M=M, pix_eps=pix_eps, **kwargs)

    return fourier2d_single_augment(img1, pix_eps, M=M, norm_ord=norm_ord, **kwargs)


def fourier2d_single_augment(img, amp_eps, M, strategy="aug", indiv_channels=False, norm_ord=np.inf, model=None, y=None, network_decoding=True, ft_symmetric=True):
    b, c, h, w = img.shape
    aug_shape = (b, c if indiv_channels else 1, h, w)
    amp_eps = eps_pixinf2ftinf(amp_eps, M)
    M_batch = torch.FloatTensor(M).unsqueeze(0).unsqueeze(0).repeat(aug_shape[0], aug_shape[1], 1, 1)
    # assert np.max(np.abs(M_batch[0][0] - M)) == 0

    if "aug" in strategy:
        M_batch = M_batch.numpy()
        amp_noise = noise(aug_shape, randomise="deterministic" not in strategy, ft_symmetric=ft_symmetric) * M_batch * amp_eps
        pha_noise = noise(aug_shape, randomise="deterministic" not in strategy, ft_symmetric=ft_symmetric, conjugate=True) * M_batch * np.pi
        if norm_ord != np.inf:
            amp_noise = zerocentered_pert_reproject(amp_noise, amp_eps, M_batch, norm_ord, not indiv_channels)
            pha_noise = zerocentered_pert_reproject(pha_noise, np.pi, M_batch, norm_ord, not indiv_channels)
        return img, polar2z(amp_noise, pha_noise), amp_noise, pha_noise

    if strategy == "robust":
        z_bounds = polar2cart_zerocentered_amp_bounds(M_batch.numpy()*amp_eps)
        return img, z_bounds

    if strategy == "adv":
        # fourier_pgd_attack is primarily written/used to be able to construct pgd attack in norms other than l_inf in fourier domain
        adv_z = fourier_pgd_zerocentered_amp_attack(model, img, y, amp_eps=amp_eps, mask=M_batch, channel_uniform_eps=not indiv_channels)
        return img, adv_z

    if strategy == "adv_LFBA":
        adv_z = LFBA_zero_centered_attack(model, img, y, amp_eps, M_batch, network_decoding)
        return img, adv_z

    raise NotImplementedError(f"Strategy {strategy} not implemented for fourier augmentaion")


def fourier2d_pair_augment(img1, img2, M, pix_eps=1, strategy="aug", indiv_channels=False, model=None, y=None, y2=None, **kwargs):
    # assert -0.5 <= pix_eps and pix_eps <= 1.5
    amp1, pha1, z1 = get_dft(img1)
    amp2, _, _ = get_dft(img2)

    pix_eps_M = pix_eps*M
    z1_2 = polar2z(amp1 * (1-pix_eps_M) + amp2 * pix_eps_M, pha1)

    if "aug" in strategy:
        b, c, w, h = img1.shape
        pix_eps_M_rand = pix_eps_M * np.random.random((b, c if indiv_channels else 1, w, h))
        aug_z1_2 = polar2z(amp1 * (1-pix_eps_M_rand) + amp2 * pix_eps_M_rand, pha1)
        return z1, aug_z1_2, z1_2, None

    if strategy == "robust":
        return z1, None, z1_2, None

    if strategy == "adv":
        z_adv = torch.zeros((img1.shape[0], 1, 1, 1))
        adv_z1_2 = pgd_attack(model, z_adv, y, x_lims=[torch.zeros((img1.shape[0], 1, 1, 1)), pix_eps*torch.ones((img1.shape[0], 1, 1, 1))], x0=z1, x1=z1_2)
        return z1, adv_z1_2, z1_2, None
    raise NotImplementedError(f"Strategy {strategy} not implemented for fourier augmentaion")


def fourier2d_kernel_augment(img, kernel_dft_range=None, kernel_config=None, strategy="aug", model=None, y=None, **kwargs):
    if kernel_dft_range is None:
        kernel_dft_range = get_kernel_range(img.shape, **kernel_config)

    dft_x = np.fft.fft2(img, axes=(img.ndim-2, img.ndim-1))/(img.shape[-2]*img.shape[-1])
    z = dft2ztorchf(dft_x)

    if "aug" in strategy:
        dft_k_l, dft_k_u = kernel_dft_range
        dft_k = dft_k_l + np.random.random(dft_k_l.shape)*(dft_k_u - dft_k_l)
        dft_aug = dft_x * dft_k
        return z, dft2ztorchf(dft_aug), np.abs(dft_k_l), np.abs(dft_k_u)

    if strategy == "robust":
        amp_l, amp_u, pha_l, pha_u = mult_bound(dft_x, kernel_dft_range)
        return z, polar2cart_bounds(amp_l, amp_u, pha_l, pha_u)
    # if strategy == "adv":
    #     adv_z = fourier_pgd_attack(model, z, img, amp_noise, pha_noise, y, amp_eps=amp_eps, pha_eps=pha_eps)
    #     return adv_z
    raise NotImplementedError(f"Strategy {strategy} not implemented for fourier augmentaion")


if __name__ == "__main__":
    shape = (3, 8, 8)
    x = torch.rand((2, *shape))
    z, z_aug = fourier2d_augment(x, {'freq_mask_range': (0, 4)}, amp_eps=0.05)[:2]
    print(z.shape, z_aug.shape)
