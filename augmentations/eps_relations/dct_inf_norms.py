# Author: Alessandro De Palma
import torch
import numpy as np
from scipy.fft import dctn, idctn


def compute_eps(x):
    # computes the infinity norm as max of complex amplitude
    return np.linalg.norm(x, ord=float('inf'))


def eps_dct(x):
    # dividing by N as in our derivation the DFT normalizes by 1/N whereas torch keeps it constant
    return compute_eps(dctn(x, axes=[0], type=2, norm='ortho'))


def eps_idct(x):
    return compute_eps(idctn(x, axes=[0], type=2, norm='ortho'))


def generate_real_vector(eps, n):
    return (torch.rand(n)-0.5)*2*eps


def get_max_row_l1_norm_idct(mask):
    n, max_l1 = mask.shape[0], 0.
    for i in range(n):
        c_l1 = mask[0] / np.sqrt(n)
        for j in range(1, n):
            c_l1 += mask[j] * np.abs(np.cos(np.pi / n * (i + 0.5) * j)) * np.sqrt(2/n)
        if c_l1 > max_l1:
            max_l1 = c_l1
    return max_l1
    

if __name__ == "__main__":

    tests = int(1e2)
    n = 50
    eps = 10.

    # checking the fft and ifft space points 
    x = generate_real_vector(eps, n)
    print(x.shape, x[0], '\n')

    mask = np.array([1]*n)
    factor = get_max_row_l1_norm_idct(mask)  # (2*np.sqrt(n**2))  # 2*(np.sqrt(2)*(n-1)+1)/np.sqrt(n)

    max_dct_norm = 0.
    for _ in range(tests):
        c_dct_norm = eps_dct(generate_real_vector(eps, n))
        if c_dct_norm > max_dct_norm:
            max_dct_norm = c_dct_norm
    print("max random DCT norm: {} \t (should be <= eps * sqrt(n): {})".format(max_dct_norm, eps * np.sqrt(n)))
    worst_x = np.ones(n)*eps
    print("DCT norm of worst-case: {} \t (should be = eps * sqrt(n): {}))".format(eps_dct(worst_x), eps * np.sqrt(n)))
    worst_x = -np.ones(n)*eps
    print("DCT norm of worst-case: {} \t (should be = eps * sqrt(n): {}))".format(eps_dct(worst_x), eps * np.sqrt(n)))

    print(f"\n{'='*80}\nChecking how eps_inf pert for all DCT space translates to eps_inf in pixel space")
    mask = np.ones(n)
    K = get_max_row_l1_norm_idct(mask)
    max_idct_norm = 0.
    for _ in range(tests):
        c_idct_norm = eps_idct(generate_real_vector(eps/K, n))
        if c_idct_norm > max_idct_norm:
            max_idct_norm = c_idct_norm
    print("in the following K is the max l1 norm amongst the (masked) rows of the IDCT matrix")
    print("max random IDCT norm: {} \t (should be <= eps_hat = {})".format(max_idct_norm, eps))
    worst_x = np.ones(n)*eps/K
    print("IDCT norm of worst-case: {} \t (should be = eps_hat = {})".format(eps_idct(worst_x), eps))

    print(f"\n{'='*80}\nChecking how eps_inf pert in d DCT space translates to eps_inf in pixel space")
    d = 5
    mask = torch.LongTensor(([1]*d + [0]*(n-d)))
    K = get_max_row_l1_norm_idct(mask)
    print(f"d: {d}, K: {K}")
    max_idct_norm = 0.
    for _ in range(tests):
        c_idct_norm = eps_idct(mask * generate_real_vector(eps/K, n))
        if c_idct_norm > max_idct_norm:
            max_idct_norm = c_idct_norm
    print("max random masked IDCT norm: {} \t (should be <= eps_hat = {})".format(max_idct_norm, eps))
    worst_xhat = np.ones(n)*eps/K
    print("masked IDCT norm of worst-case: {} \t (should be = eps_hat = {})".format(eps_idct(mask * worst_xhat), eps))

    mask = torch.LongTensor(([0]*d + [1]*(n-d)))
    K = get_max_row_l1_norm_idct(mask)
    max_idct_norm = 0.
    for _ in range(tests):
        c_idct_norm = eps_idct(mask * generate_real_vector(eps/K, n))
        if c_idct_norm > max_idct_norm:
            max_idct_norm = c_idct_norm
    print("max random masked IDCT norm: {} \t (should be <= eps_hat = {})".format(max_idct_norm, eps))
    worst_xhat = np.ones(n)*eps/K
    print("masked IDCT norm of worst-case: {} \t (should be = eps_hat = {})".format(eps_idct(mask * worst_xhat), eps))

    # ==> the smallest l-inf ball in the pixel space enclosing all
    # (D-dimensional and with d non-zero elements) DCT perturbations with
    # l-inf radius eps_hat, has radius eps = K * eps_hat, with K being the max l1 norm amongst the (masked) rows of the IDCT matrix

    # ==> the smallest l-inf ball in the DCT space enclosing all
    # (D-dimensional and with d non-zero elements) pixel perturbations with
    # l-inf radius eps, has radius eps_hat = sqrt(d) * eps
