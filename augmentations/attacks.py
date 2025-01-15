import torch
from torch import nn

import numpy as np
from math import exp


from .fourier2d_utils import z2polar, polar2z


def channel_standardise_eta(eta):
    eta_measure = torch.max(eta, dim=1, keepdim=True)[0]
    return eta_measure.repeat(1, eta.shape[1], 1, 1)


def zerocentered_pert_reproject(pert, eps, mask, norm_ord, channel_uniform_eps):
    if not isinstance(pert, torch.Tensor):
        pert = torch.tensor(pert)
    if norm_ord == np.inf:
        eta = torch.clamp(pert, min=-eps, max=eps)
    else:
        eta = torch.empty_like(pert)
        for c in range(eta.shape[-3]):
            maxnorm = (eps if isinstance(eps, float) else eps[0, c, 0, 0].item())/np.sqrt(eta.shape[-3])
            eta[:, c, ...] = pert[:, c, ...].renorm(p=norm_ord, dim=0, maxnorm=maxnorm)
    # print(f"\n eps: {eps}, channel norm: {torch.norm(eta[0, c, ...], p=2)}\n")

    if channel_uniform_eps:
        eta = channel_standardise_eta(eta)

    eta *= mask
    # print(f"\n eps: {eps if isinstance(eps, float) else eps.max()}, norm_{norm_ord}: {torch.norm(eta[0], p=norm_ord)}, channel_uniform: {channel_uniform_eps}, nz: {np.count_nonzero(mask)}\n")
    return eta.float()


def pert_reproject(x_adv, x, eps, mask, norm_ord, channel_uniform_eps):
    if eps is None:
        return x, np.zeros_like(x)
    diff = torch.tensor((x_adv - x) * mask)
    eta = zerocentered_pert_reproject(diff, eps, mask, norm_ord, channel_uniform_eps)
    x = x + eta
    return x, eta


def fourier_pgd_zerocentered_amp_attack(model, x, y, amp_eps, mask=1, alpha=None, iters=25, norm_ord=np.inf, channel_uniform_eps=False):
    alpha = alpha or amp_eps/3
    mask = torch.concat((mask, mask), dim=mask.ndim-2)
    amp_masked_eps = amp_eps * mask
    z_adv = (torch.ones(mask.shape) * np.random.choice([-1., 1.], mask.shape) * amp_masked_eps).float()

    x, y = x.to(model.device), y.to(model.device)
    with torch.enable_grad():
        for i in range(iters):
            model.zero_grad()
            z_adv.requires_grad = True
            logits = model(z_adv.to(model.device), c_x=x)[0]
            if i > 0 and (logits.argmax(-1) != y).sum() == len(y):
                # adversarial example found for all batch samples
                break
            loss = nn.functional.cross_entropy(logits, y)
            grad_loss_z = torch.autograd.grad(loss, [z_adv])[0]

            z_adv_ = (z_adv.detach() + alpha * mask * grad_loss_z.detach().sign())
            z_adv = zerocentered_pert_reproject(z_adv_, amp_eps, mask, norm_ord, channel_uniform_eps)
    # print("iters done", i, np.count_nonzero(mask)/len(mask)/2, np.count_nonzero(z_adv)/len(mask)/6, amp_eps, torch.max(z_adv).item(), torch.min(z_adv).item())
    return z_adv.to(torch.float)


def pgd_attack(model, x, y, eps=None, x_lims=None, x0=None, x1=None, alpha=1/255, iters=25, channel_uniform_eps=False, norm_ord=np.inf, mask=1, clamp_01=False):
    def lims_reproject(x_adv, x):
        eta = torch.clamp(x_adv, min=x_lims[0], max=x_lims[1]) - x
        if channel_uniform_eps:
            eta = channel_standardise_eta(eta)
        return x + eta

    # from torchattacks.attacks.pgd import PGD
    # with torch.enable_grad():
    #     x_adv = PGD(model, eps=eps, alpha=alpha, steps=iters, random_start=True)(x_denorm, y)  # could use if didn't have to use mask
    # return dataset.normalize(x_adv)

    assert eps is not None or x_lims is not None, "Provide at least eps or x_lims for PGD attack"
    x, y = x.to(model.device), y.to(model.device)
    if type(mask) is torch.Tensor:
        mask = mask.to(model.device)

    if eps is not None:
        alpha = eps/5
        x_adv = (x + (torch.rand_like(x) - 0.5)*2 * eps).to(model.device)
        x_adv, _ = pert_reproject(x_adv, x, eps, mask, norm_ord, channel_uniform_eps)
    else:
        assert np.isinf(norm_ord), "Can only find l_inf adv example if limits provided, otherwise specify eps"
        x_adv = x_lims[0] + torch.rand(x.shape)*(x_lims[1]-x_lims[0])
        x_adv = lims_reproject(x_adv, x)
    if clamp_01:
        x = torch.clamp(x, min=0, max=1)  # clipping say for image pixels which lie between [0,1] float normalized

    with torch.enable_grad():
        for i in range(iters):
            model.zero_grad()
            x_adv.requires_grad = True
            if x0 is not None and x1 is not None:
                # find adv example in a segment spanning x0, x1
                # (model uses segment endpts to create app. prepending layers) and x_adv is the ls alpha
                logits = model(x=x0, x1=x1, alpha=x_adv)[0]
            else:
                logits = model(x=x_adv)[0]
            correct_mask = logits.argmax(dim=1) == y
            if (correct_mask).sum() == 0:
                # adversarial example found for all batch samples
                break
            loss = nn.functional.cross_entropy(logits, y).to(model.device)
            grad_loss_x = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + alpha * correct_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1) * grad_loss_x.detach().sign()

            if eps is not None:
                x_adv, _ = pert_reproject(x_adv, x, eps, mask, norm_ord, channel_uniform_eps)
            else:
                x_adv = lims_reproject(x_adv, x)

            if clamp_01:
                x = torch.clamp(x, min=0, max=1)  # clipping say for image pixels which lie between [0,1] float normalized

        model.zero_grad()
    assert x_adv.shape == x.shape, f"{x_adv.shape} {x.shape}"

    return x_adv.to(torch.float).to(model.device)


# not used in submission
def fourier_pgd_zerocentered_amp_attack_v2(model, x, y, amp_eps, mask=1, alpha=None, iters=25, norm_ord=np.inf, channel_uniform_eps=False):
    alpha = alpha or amp_eps/5
    adv_amp = (np.random.random(x.shape)-0.5) * amp_eps * mask
    adv_pha = (np.random.random(x.shape)-0.5) * np.pi * mask  # [-pi, pi]
    z_adv = polar2z(adv_amp, adv_pha)
    x, y = x.to(model.device), y.to(model.device)
    with torch.enable_grad():
        for i in range(iters):
            z_adv.requires_grad = True
            model.zero_grad()
            logits = model(z_adv.to(model.device), c_x=x)[0]
            if i > 0 and (logits.argmax(-1) != y).sum() == len(y):
                # adversarial example found for all batch samples
                break
            loss = nn.functional.cross_entropy(logits, y)
            grad_loss_z = torch.autograd.grad(loss, [z_adv])[0]
            # loss.backward()

            z_adv_ = (z_adv.detach() + alpha * grad_loss_z.detach().sign()).numpy()  # this is z space, limits are in amp, pha
            adv_amp, adv_pha = z2polar(z_adv_)
            adv_amp = zerocentered_pert_reproject(adv_amp, amp_eps, mask, norm_ord, channel_uniform_eps)
            z_adv = polar2z(adv_amp, adv_pha)
    return z_adv.to(torch.float)


def fourier_pgd_attack(model, x, z, amp, pha, y, amp_eps, pha_eps=None, mask=1, alpha=0.5/255, iters=50, norm_ord=np.inf, channel_uniform_eps=False):
    adv_amp = (np.random.randn(*amp.shape)-0.5) * amp_eps * mask
    adv_pha = (np.random.randn(*pha.shape)-0.5) * pha_eps * mask if pha_eps else 0
    z_adv = polar2z(adv_amp, adv_pha)
    with torch.enable_grad():
        for i in range(iters):
            z_adv.requires_grad = True
            model.zero_grad()
            logits = model(z_adv.to(model.device), c_x=x.to(model.device))[0]
            if (logits.argmax(-1) != y).sum() == len(y):
                # adversarial example found for all batch samples
                break
            loss = nn.functional.cross_entropy(logits, y).to(model.device)
            grad_loss_z = torch.autograd.grad(loss, [z_adv])[0]
            # loss.backward()

            z_adv_ = (z_adv.detach() + alpha * grad_loss_z.detach().sign()).numpy()  # this is z space, limits are in amp, pha
            # todo: symmetrify z_adv here
            adv_amp, adv_pha = z2polar(z_adv_)
            adv_amp, _ = pert_reproject(adv_amp, amp, amp_eps, mask, norm_ord, channel_uniform_eps, False)
            adv_pha, _ = pert_reproject(adv_pha, pha, pha_eps, mask, norm_ord, channel_uniform_eps, False)
            z_adv = polar2z(adv_amp, adv_pha)
            # print(torch.abs(z_adv-z_adv_).mean())
    return z_adv.to(torch.float)


# popular frequency space attacks

# 1. LFBA (https://arxiv.org/pdf/2402.15653v2.pdf, exact implementation not available, so reimplimented Algorithm 1
# based on Simulated Annealing algo from https://machinelearningmastery.com/simulated-annealing-from-scratch-in-python/)
def LFBA_zero_centered_attack(model, x, y, amp_eps, mask, network_decoding, temp_range=[0.8, 10], iters=7, alpha=0.1, freq_transform="fourier"):
    def objective(logits, x_adv=None):
        return -nn.functional.cross_entropy(logits, y).to(model.device)

    # generate an initial point
    if freq_transform == "fourier":
        mask = torch.concat((mask, mask), dim=mask.ndim-2)
    amp_masked_eps = 2*amp_eps * mask
    best_z = ((torch.rand(mask.shape)-0.5) * amp_masked_eps).float()

    inference_kwargs = {}
    if freq_transform == "fourier":
        inference_kwargs['c_x'] = x.to(model.device)
    else:
        best_z = x + best_z

    # evaluate the initial point, get output from here, put constraint on that
    y = y.to(model.device)
    logits = model(best_z.to(model.device), **inference_kwargs)[0]

    best_obj = objective(logits)

    # current working solution
    curr_z, curr_obj = best_z, best_obj

    # run the algorithm for a range of decreasing temperatures
    temp = temp_range[1]
    while temp > temp_range[0]:
        for i in range(iters):
            # take a step
            cand_z = (curr_z + alpha * (torch.rand(curr_z.shape)-0.5) * amp_masked_eps).float()
            # evaluate candidate point
            logits, x_adv = model(cand_z.to(model.device), **inference_kwargs)[:2]
            incorrect_mask = logits.argmax(-1) != y
            if (incorrect_mask).sum() == len(y):
                # adversarial example found for all batch samples
                break
            cand_obj = objective(logits)
            # check for and store new best solution
            if cand_obj < best_obj:
                # store new best point that reduces the objective
                best_z, best_obj = cand_z, cand_obj
            # difference between candidate and current point evaluation
            diff = cand_obj - curr_obj
            # calculate temperature for current epoch
            t = temp / float(i + 1)
            # calculate metropolis acceptance criterion
            metropolis = exp(-diff / t)
            # check if we should keep the new point
            if diff < 0 or np.random.rand() < metropolis:
                # store the new current point
                curr_z, curr_obj = cand_z, cand_obj
        temp *= (1-alpha)
    return torch.FloatTensor(best_z)


# 2. FTrojan (https://arxiv.org/pdf/2111.10991.pdf, adopted from provided implementation:
# https://github.com/SoftWiser-group/FTrojan)
# def poison(x_train, window_size, magnitude, pos_list):
#     if x_train.shape[0] == 0:
#         return x_train

#     # transfer to frequency domain
#     x_train = DFT(x_train, window_size)  # (idx, ch, w, h)

#     # plug trigger frequency
#     for i in range(x_train.shape[0]):
#         for ch in x_train.shape[1]:  # BS, C, W, H
#             for w in range(0, x_train.shape[2], window_size):
#                 for h in range(0, x_train.shape[3], window_size):
#                     for pos in pos_list:
#                         x_train[i][ch][w + pos[0]][h + pos[1]] += magnitude

#     # x_train = IDFT(x_train, window_size)  # (idx, w, h, ch)
#     return x_train


if __name__ == "__main__":
    print(polar2z(2, np.random.random((1))*2*np.pi, cat_torchf=False))
