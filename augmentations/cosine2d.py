import torch

import numpy as np
from scipy.fft import dctn

from .fourier2d_utils import noise, get_freq_mask, eps_pixinf2ftinf
from .attacks import LFBA_zero_centered_attack, pgd_attack


def dct2d_augment(img, eps, M=None, freq_mask_config=None, strategy="aug", indiv_channels=False, norm_ord=np.inf, model=None, y=None, network_decoding=True, **kwargs):
    b, c, h, w = img.shape
    if M is None:
        M = get_freq_mask((c, h, w), **freq_mask_config)
    eps = 2*eps_pixinf2ftinf(eps, M, "cosine")
    aug_shape = (b, c if indiv_channels else 1, h, w)
    M_batch = torch.FloatTensor(M).unsqueeze(0).unsqueeze(0).repeat(aug_shape[0], aug_shape[1], 1, 1)
    assert np.max(np.abs(M_batch[0][0].numpy() - M)) == 0

    img_dct = dctn(img.numpy(), axes=[2, 3], norm='ortho')

    if "aug" in strategy:
        M_batch = M_batch.numpy()
        pert = noise(aug_shape, randomise="deterministic" not in strategy, ft_symmetric=True) * M_batch * eps
        return torch.FloatTensor(img_dct), torch.FloatTensor(img_dct + pert)

    img_dct = torch.FloatTensor(img_dct)

    if strategy == "robust":
        masked_eps = eps*M
        z_bounds = torch.stack([img_dct-masked_eps, img_dct+masked_eps])
        return img, z_bounds.to(torch.float)

    if strategy == "adv":
        adv_z = pgd_attack(model, img_dct, y, eps, channel_uniform_eps=not indiv_channels, norm_ord=norm_ord, mask=M_batch, clamp_01=False)
        return img, adv_z

    if strategy == "adv_LFBA":
        adv_z = LFBA_zero_centered_attack(model, img_dct, y, eps, M_batch, network_decoding, freq_transform="cosine")
        return torch.FloatTensor(img_dct), adv_z

    raise NotImplementedError(f"Strategy {strategy} not implemented for fourier augmentaion")
