# Author: Alessandro De Palma
import torch
import numpy as np


def compute_eps(x):
    # computes the infinity norm as max of complex amplitude
    return x.norm(p=float('inf'), dim=-1)


def eps_dct(x):
    return compute_eps(torch.fft.fft(x, dim=-1) / x.shape[-1])


def eps_idct(x):
    return compute_eps(torch.fft.ifft(x, dim=-1) * x.shape[-1])


def generate_real_vector(eps, n):
    # generates an n-dimensional real vector of infinity norm = eps
    return (torch.rand(n)-0.5)*2*eps  # torch.stack([torch.rand(n)*eps, torch.zeros(n)], dim=1)


def generate_complex_vector(eps, n):
    # generates an n-dimensional complex vector of infinity norm = eps
    # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
    # radius = torch.sqrt(torch.rand(n)) * eps
    # angle = torch.rand(n) * 2 * pi
    # return radius * torch.cos(angle) + 1j * radius * torch.sin(angle)
    return (torch.rand(n)-0.5 + 1j*(torch.rand(n)-0.5))*2*eps


if __name__ == "__main__":

    tests = int(1e2)
    n = 50
    eps = 10.

    # checking the fft and ifft space points 
    x = generate_real_vector(eps, n)
    print(x.shape, x[0])
    x = generate_complex_vector(eps, n)
    print(x.shape, x[0])

    print(f"\n{'='*80}\nChecking how eps_inf pert in pixel space translates to eps_inf in DFT cartesian space")
    max_dft_norm = 0.
    for _ in range(tests):
        c_dft_norm = eps_dct(generate_real_vector(eps, n))
        if c_dft_norm > max_dft_norm:
            max_dft_norm = c_dft_norm
    print(f"max random DFT norm: {max_dft_norm} \t (should be <= eps={eps})")  # should <= n*eps
    worst_x = torch.ones(n)*eps
    print(f"DFT norm of worst-case: {eps_dct(worst_x)} \t (should be = eps={eps})")  # should be = n*eps

    print(f"\n{'='*80}\nChecking how eps_inf pert for all DFT cartesian space translates to eps_inf in pixel space")
    max_idft_norm = 0.
    for _ in range(tests):
        c_idft_norm = eps_idct(generate_complex_vector(eps/n, n))
        if c_idft_norm > max_idft_norm:
            max_idft_norm = c_idft_norm
    print(f"max random IDFT norm: {max_idft_norm} \t (should be <= eps={n*eps})")  # should <= n * eps

    worst_x = torch.ones(n)*eps/n
    print(f"IDFT norm of worst-case: {eps_idct(worst_x)} \t (should be = eps={n*eps})")  # should = n * eps
    print(f"IDFT norm of worst-case: {eps_idct(1j*worst_x)} \t (should be = eps={n*eps})")  # should = n * eps

    print(f"\n{'='*80}\nChecking how eps_inf pert in d DFT cartesian space translates to eps_inf in pixel space")
    d = int(0.25*n)
    print("perturbation dimension:", d)
    mask = np.array([1]*d + [0]*(n-d))
    max_idft_norm = 0.
    for _ in range(tests):
        np.random.shuffle(mask)
        c_idft_norm = eps_idct(generate_complex_vector(eps/d, n)*mask)
        if c_idft_norm > max_idft_norm:
            max_idft_norm = c_idft_norm
    print(f"max random masked IDFT norm: {max_idft_norm} \t (should be <= eps={d*eps})")  # should <= d * eps

    worst_x = torch.ones(n)*eps*mask/d
    print(f"IDFT norm of worst-case: {eps_idct(worst_x)} \t (should be = eps={d*eps})")  # should = d * eps
    print(f"IDFT norm of worst-case: {eps_idct(1j*worst_x)} \t (should be = eps={d*eps})")  # should = d * eps

    # ==> the smallest l-inf ball in the pixel space enclosing all
    # (D-dimensional and with d non-zero elements) DFT perturbations with
    # l-inf radius eps_hat, has radius eps = d * eps_hat

    # ==> the smallest l-inf ball in the DFT space enclosing all
    # (D-dimensional and with d non-zero elements) pixel perturbations with
    # l-inf radius eps, has radius eps_hat = eps
