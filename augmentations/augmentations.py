import torch

import numpy as np

from .fourier2d import get_freq_mask, get_kernel_range, fourier2d_addmult_augment, fourier2d_kernel_augment, get_dft, pgd_attack
from .attacks import zerocentered_pert_reproject
from .cosine2d import dct2d_augment
from utils import Scheduler


class InputProcessor:
    def __init__(self, data_info, aug_cfg={}):
        self.aug_domain = aug_cfg.get("domain", "pixel")

    def process(self, x, x1=None):
        if self.aug_domain == "fourier" and x1 is not None:
            return get_dft(x, only_cart=True), get_dft(x1, only_cart=True)
        return x, x1


class InputAugmentor(InputProcessor):
    def __init__(self, data_info, aug_cfg={}):
        super().__init__(data_info, aug_cfg)
        shape, self.std_dev = data_info["shape"], data_info["std"]

        self.freq_mask, self.kernel_dft_range = None, None
        if self.aug_domain in ["fourier", "cosine"]:
            if aug_cfg["pert_model"] == "spatial_conv":
                self.kernel_dft_range = get_kernel_range(shape, **aug_cfg["kernel_config"])
            else:
                self.freq_mask = get_freq_mask(shape, **aug_cfg["fourier_mask"], resolution_multiple=aug_cfg.get("resolution_multiple", 1))
        elif self.aug_domain == "pixel":
            assert aug_cfg.get("pert_model", "additive") in ["additive"], f"Pert model {aug_cfg['pert_model']} not supported for pixel domain"

        self.kernel_cfg, self.pix_eps = aug_cfg.get("kernel_config", None), aug_cfg.get("pix_eps", None)
        self.pert_factor, self.pert_eps = 1, self.pix_eps
        self.pert_scheduler = Scheduler(**aug_cfg["pert_scheduling"]) if aug_cfg.get("pert_scheduling", None) else None
        self.aug_cfg = {k: v for k, v in aug_cfg.items() if k not in ["domain", "fourier_mask", "kernel_config", "pix_eps", "spec", "pert_scheduling", "bounds_config", "minimum_robust_weight"]}  # spec being ball/segment handled in data

    def augment_and_process(self, epoch, x, x1=None, **kwargs):
        kernel_config, kernel_dft_range = None, self.kernel_dft_range

        if epoch is not None and self.pert_scheduler is not None:
            self.pert_factor = self.pert_scheduler(epoch)
            if self.aug_domain == "fourier" and self.aug_cfg["pert_model"] == "spatial_conv":
                kernel_config = self.kernel_config.copy()
                kernel_config["kernel_params_range"][1], kernel_dft_range = self.pert_factor * self.kernel_config["kernel_params_range"][1], None
            else:
                self.pert_eps = self.pert_factor * self.pix_eps

        if self.aug_domain == "fourier":
            if self.aug_cfg["pert_model"] == "spatial_conv":
                return fourier2d_kernel_augment(x, kernel_dft_range=kernel_dft_range, kernel_config=kernel_config, **self.aug_cfg, **kwargs)
            else:
                return fourier2d_addmult_augment(x, pix_eps=self.pert_eps, M=self.freq_mask, img2=x1, **self.aug_cfg, **kwargs)
        elif self.aug_domain == "pixel":
            # eps = (self.pert_eps / torch.tensor(self.std_dev)).view(1, -1, 1, 1)  # if normalizing in dataset loading, not in network, which we never do
            eps = torch.tensor(self.pert_eps).repeat(1, 3, 1, 1)
            return pixel_pair_augment(x, eps, x1, **self.aug_cfg, **kwargs) if x1 is not None else pixel_single_augment(x, eps, **self.aug_cfg, **kwargs)
        elif self.aug_domain == "cosine":
            return dct2d_augment(x, eps=self.pert_eps, M=self.freq_mask, **self.aug_cfg, **kwargs)
        raise NotImplementedError("{self.aug_domain} domain not supported for augmentation")


def pixel_single_augment(img, eps, strategy="aug", indiv_channels=False, M=1, norm_ord=np.inf, model=None, y=None, **kwargs):
    if img.ndim == 3:
        img = img.unsqueeze(0)
    eps *= M
    if strategy == "aug":
        b, c, h, w = img.shape
        aug = (torch.rand((b, c if indiv_channels else 1, h, w))-0.5)*2 * eps
        aug = zerocentered_pert_reproject(aug, eps, M, norm_ord, not indiv_channels)
        img_aug = torch.clamp(img + aug, min=0, max=1)
        return img, img_aug

    if strategy == "robust":
        img_bounds = torch.clamp(torch.stack([img-eps, img+eps]), 0, 1)
        return img, img_bounds

    if strategy == "adv":
        return img, pgd_attack(model, img, y, eps=eps.to(model.device), norm_ord=norm_ord, mask=M, channel_uniform_eps=not indiv_channels, clamp_01=True)
    raise NotImplementedError(f"Strategy {strategy} not implemented for fourier augmentaion")


def pixel_pair_augment(img1, eps, img2, strategy="aug", indiv_channels=False, M=1, norm_ord=np.inf, model=None, y=None, y2=None, **kwargs):
    # assert -0.5 <= eps and eps <= 1.5, print("eps: {eps}")
    if strategy == "aug":
        b, c, h, w = img1.shape
        noise_mul = M * (torch.rand((b, c if indiv_channels else 1, h, w)) * eps)
        img_min, img_max = np.minimum(img1, img2), np.maximum(img1, img2)
        aug1 = torch.clamp(img1 + noise_mul * (img2-img1), img_min, img_max)
        aug2 = torch.clamp(img2 + noise_mul * (img1-img2), img_min, img_max)
        return img1, aug1, img2, aug2

    # fix the following
    aug12_max, aug21_max = img2, img1
    if eps.min() != 1:
        aug12_max = np.maximum(img1 + eps * M * (img2 - img1), -1)
        aug21_max = np.maximum(img2 + eps * M * (img1 - img2), -1)

    if strategy == "robust":
        return img1, torch.clamp(torch.stack([img1, aug12_max]), 0, 1), img2, torch.clamp(torch.stack([img2, aug21_max]), 0, 1)

    if strategy == "adv":
        adv_z1 = pgd_attack(model, img1, y, x_lims=(img1, aug12_max), norm_ord=norm_ord, channel_uniform_eps=not indiv_channels, clamp_01=True)
        adv_z2 = pgd_attack(model, img2, y2 or y, x_lims=(img2, aug21_max), norm_ord=norm_ord, channel_uniform_eps=not indiv_channels, clamp_01=True)
        return img1, adv_z1, img2, adv_z2
    raise NotImplementedError(f"Strategy {strategy} not implemented for pixel augmentaion")
