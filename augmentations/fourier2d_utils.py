import torch

import numpy as np
from functools import partial


# Mask utils
def set_radius(M, r, v, res_mult=1):
    h, w = M.shape
    ch, cw = h//2, w//2

    if r == 0:
        M[ch:ch+1, cw:cw+1] = v
        return

    if res_mult == 1:
        M[ch-r: ch+r+1, cw-r: cw+r+1] = v
    else:
        M[ch: ch+r+1, cw: cw+r+1] = v


def get_freq_mask(shape, freq_mask_range=(0, 1), allow_inner_freqs=True, resolution_multiple=1, shift=True):
    if freq_mask_range[1] is None:
        M = np.ones(shape[-2:]) if allow_inner_freqs else np.zeros(shape[-2:])
        return M

    R_l, R_h = int(freq_mask_range[0]), int(freq_mask_range[1])
    if R_l < 0:
        R_l = shape[-1] // 2 + 1 + R_l
    if R_h < 0:
        R_h = shape[-1] // 2 + 2 + R_h

    M = np.zeros(shape[-2:])
    h, w = M.shape
    for R_ in range(R_l, R_h):
        M_ = np.zeros(M.shape)
        set_radius(M_, R_, 1, resolution_multiple)
        if R_ > 0:
            set_radius(M_, R_-1, 0, resolution_multiple)
        M += M_
    M = M.astype(bool).astype(float)
    if not allow_inner_freqs:
        M = 1 - M
    assert np.max(M) == 1, np.max(M)
    return np.fft.ifftshift(M, axes=(0, 1)) if shift else M


# bound utils
def polar2cart_zerocentered_amp_bounds(amp_abs, cat_torchf=True):
    real_l, real_u = -amp_abs, amp_abs
    imag_l, imag_u = -amp_abs, amp_abs

    if not cat_torchf:
        return real_l, real_u, imag_l, imag_u

    z_l = np.concatenate((real_l, imag_l), axis=real_l.ndim-2)
    z_u = np.concatenate((real_u, imag_u), axis=real_u.ndim-2)
    z_bounds = np.stack((z_l, z_u))
    assert np.max(z_u-z_l) == 2*np.max(amp_abs), f"{np.min(z_u-z_l), np.max(z_u-z_l)}"
    return torch.from_numpy(z_bounds).to(torch.float)


def polar2cart_bounds(abs_l, abs_u, pha_l, pha_u, assert_inside_z=None, cat_torchf=True):
    r1, i1 = polar2z(abs_l, pha_l, cat_torchf=False)
    r2, i2 = polar2z(abs_l, pha_u, cat_torchf=False)
    r3, i3 = polar2z(abs_u, pha_u, cat_torchf=False)
    r4, i4 = polar2z(abs_u, pha_l, cat_torchf=False)
    real_l = np.minimum(np.minimum(r1, r2), np.minimum(r3, r4))
    real_u = np.maximum(np.maximum(r1, r2), np.maximum(r3, r4))
    imag_l = np.minimum(np.minimum(i1, i2), np.minimum(i3, i4))
    imag_u = np.maximum(np.maximum(i1, i2), np.maximum(i3, i4))
    # pha must be in [-pi, pi]
    pha_l, pha_u = wrap_angle(pha_l), wrap_angle(pha_u)
    # pha_l < np.pi/2 and pha_u > np.pi/2: imag_u = abs_u  # crossing the pos y-axis
    imag_u = np.where(np.bitwise_and(pha_l < np.pi/2, pha_u > np.pi/2), abs_u, imag_u)
    # pha_l < -np.pi/2 and pha_u > -np.pi/2: imag_l = -abs_u  # crossing the neg y-axis
    imag_l = np.where(np.bitwise_and(pha_l < -np.pi/2, pha_u > -np.pi/2), -abs_u, imag_l)
    # pha_l < 0 and pha_u > 0: real_u = abs_u  # crossing the pos x-axis
    real_u = np.where(np.bitwise_and(pha_l < 0, pha_u > 0), abs_u, real_u)
    # pha_l < np.pi and pha_u < -np.pi: real_l = -abs_u  # crossing the neg
    real_l = np.where(np.bitwise_and(pha_l < np.pi, pha_u < -np.pi), -abs_u, real_l)

    tol = 1e-6
    assert np.max(real_u-real_l) > -tol, f"{(real_u-real_l).max()}"
    assert np.max(imag_u-imag_l) > -tol, f"{(imag_u-imag_l).max()}"

    if not cat_torchf:
        return real_l, real_u, imag_l, imag_u

    z_l = np.concatenate((real_l, imag_l), axis=real_l.ndim-2)
    z_u = np.concatenate((real_u, imag_u), axis=real_u.ndim-2)
    z_bounds = np.stack((z_l, z_u))
    tol = 1e-3
    if assert_inside_z is not None:
        assert torch.min(assert_inside_z-z_l) >= -tol and torch.max(assert_inside_z-z_u) <= tol, f"{torch.min(assert_inside_z-z_l)}!>0 {torch.max(assert_inside_z-z_u)}!<0"
    assert np.min(z_u-z_l) >= -tol, f"{np.min(z_u-z_l)}"
    return torch.from_numpy(z_bounds).to(torch.float)


def mult_bound(z1, z2_bounds):
    z1_a, z1_phi = np.abs(z1), unwrap_angle(np.angle(z1))
    z2_l, z2_u = z2_bounds
    z2_a_l, z2_phi_l, z2_a_u, z2_phi_u = np.abs(z2_l), np.angle(z2_l), np.abs(z2_u), np.angle(z2_u)
    z_a_l, z_a_u = z1_a*z2_a_l, z1_a*z2_a_u
    z_phi_l, z_phi_u = z1_phi + z2_phi_l, z1_phi + z2_phi_u
    return z_a_l, z_a_u, z_phi_l, z_phi_u


# eps conversions
def get_max_row_l1_norm_idct(mask):
    n, max_l1 = mask.shape[0], 0.
    for i in range(n):
        c_l1 = mask[0] / np.sqrt(n)
        for j in range(1, n):
            c_l1 += mask[j] * np.abs(np.cos(np.pi/n * (i+0.5)*j)) * np.sqrt(2/n)
        if c_l1 > max_l1:
            max_l1 = c_l1
    return max_l1


def eps_pixinf2ftinf(pix_eps, M, domain="fourier"):
    nz, scale = np.count_nonzero(M), 1
    if domain == "fourier":
        scale = nz ** 0.5
    elif domain == "cosine":
        scale = get_max_row_l1_norm_idct(M[0])
        scale = scale ** 2
    else:
        raise NotImplementedError()
    return pix_eps/scale


def eps_pixl22ft(pix_eps_l2, M, target_norm=None):
    def linf2l2(eps_inf, shape):
        return eps_inf * np.sqrt(np.prod(shape[-2:])/np.pi)

    def l22linf(eps_l2, shape=None, ndim=None):
        if shape is not None:
            return eps_l2 / np.sqrt(np.prod(shape[-2:])/np.pi)
        if ndim == 1:
            return eps_l2
        return eps_l2 / np.sqrt(ndim / np.pi)

    nz = np.count_nonzero(M)

    ft_eps_l2 = pix_eps_l2 / np.sqrt(np.prod(M.shape[-2:]))  # Parseval's theorem
    if target_norm is None or target_norm == 2:
        return ft_eps_l2, nz

    if target_norm == np.inf:
        ft_eps_inf = l22linf(ft_eps_l2, ndim=nz)
        return ft_eps_inf, nz
    raise NotImplementedError(f"target_norm {target_norm} not supported")


# kernel utils
def get_kernel_range(shape, kernel_type, kernel_params_range):
    if kernel_type == "gaussian":
        kernel_fn = partial(gaussian_kernel, N=shape[-1])
    else:
        raise NotImplementedError(f"Kernel {kernel_type} not supported")

    k_l, k_u = [kernel_fn(**params) for params in kernel_params_range]
    dft_k_lp, dft_k_up = [np.fft.fft2(np.fft.ifftshift(k, axes=(k.ndim-2, k.ndim-1)), axes=(k.ndim-2, k.ndim-1)) for k in [k_l, k_u]]
    return np.minimum(dft_k_lp, dft_k_up), np.maximum(dft_k_lp, dft_k_up)


def gaussian_kernel(N, sigma):
    """ creates gaussian kernel with side length `l` and a sigma of `sig` """
    ax = np.linspace(-(N - 1) / 2., (N - 1) / 2., N)
    gauss = np.exp(-0.5 * np.square(ax) / (np.square(sigma) + 1e-6))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


# fourier utils
def dft2ztorchf(dft):
    r, i = np.real(dft), np.imag(dft)
    z = np.concatenate((r, i), axis=r.ndim-2)
    return torch.from_numpy(z).to(torch.float)


def polar2z(amp, phase, cat_torchf=True):
    dft = amp * (np.e ** (1j * wrap_angle(phase)))
    if cat_torchf:
        return dft2ztorchf(dft)
    return np.real(dft), np.imag(dft)


def z2polar(z):
    real, imag = z[..., :z.shape[-2]//2, :], z[..., z.shape[-2]//2:, :]
    dft = real + 1j * imag
    return np.abs(dft), unwrap_angle(np.angle(dft))


def get_dft(img, only_cart: bool = False):
    assert img.shape[-2] == img.shape[-1], f"{img.shape}, {img.min()} {img.max()}"
    dft = np.fft.fft2(img, axes=(img.ndim-2, img.ndim-1)) / (img.shape[-2]*img.shape[-1])
    if only_cart:
        return dft2ztorchf(dft)
    return np.abs(dft), unwrap_angle(np.angle(dft)), dft2ztorchf(dft)


def unwrap_angle(phi):
    return phi % (2*np.pi)


def wrap_angle(phi):
    return (phi + np.pi) % (2*np.pi) - np.pi


# [0,1] noise
def noise(shape, randomise=True, ft_symmetric=False, **kwargs):
    if not randomise:
        return np.ones(shape)
    if not ft_symmetric:
        return (np.random.random(shape)-0.5)*2
    return symmetric_noise(shape, randomise, **kwargs)


def symmetric_noise(shape, randomise=True, conjugate=False):
    # np_func = np.random.random if randomise else np.ones
    c = -1 if conjugate else 1
    central = (np.random.random((*shape[:-2], shape[-2]//2, 1))-0.5)*2 if randomise else np.ones((*shape[:-2], shape[-2]//2, 1))
    central = np.concatenate((central, (np.random.random((*shape[:-2], 1, 1))-0.5)*2 if randomise else np.ones((*shape[:-2], 1, 1)), c*np.flip(central, axis=central.ndim-2)), axis=central.ndim-2)
    noise = (np.random.random((*shape[:-1], shape[-1]//2))-0.5)*2 if randomise else np.ones((*shape[:-1], shape[-1]//2))
    noise = np.concatenate((noise, central, c*np.flip(np.flip(noise, axis=noise.ndim-1), axis=noise.ndim-2)), axis=noise.ndim-1)
    noise = np.fft.ifftshift(noise, axes=(noise.ndim-2, noise.ndim-1))
    return noise
