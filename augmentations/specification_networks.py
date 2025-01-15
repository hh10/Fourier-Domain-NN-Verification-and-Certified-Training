import torch
from torch import nn

import numpy as np
from scipy.fft import idct

from networks.network_impl import ChannelBroadcaster


def interpolation_network(pt1, pt2):
    from data.augmentation_bp_utils import AddConstant, MulConstant
    diff = pt2 - pt1
    return nn.Sequential(MulConstant(diff), AddConstant(pt1))


bias = False


class IDFT(nn.Module):
    def __init__(self, N: int, pert_model="additive", C=3, resolution_multiple=1, concat_v=True, real_signal=True, clipping_net_type="default", indiv_channels=False, device="cpu"):
        super().__init__()
        self.N, self.C, self.resolution_multiple, self.indiv_channels = N, C, resolution_multiple, indiv_channels
        IW_full = torch.from_numpy(IDFT_matrix(resolution_multiple*N))
        IW = IW_full[:N, :N]
        if resolution_multiple > 1:
            self.IW_1N, self.IW_N1, self.IW_NN = IW_full[:N, -N:], IW_full[-N:, :N], IW_full[-N:, -N:]
        IW_r, IW_i = torch.real(IW).to(torch.float), torch.imag(IW).to(torch.float)

        self.real_signal = real_signal and resolution_multiple == 1
        init = self.init_concat if concat_v else self.init_non_concat
        init(N, IW_r, IW_i)
        assert hasattr(self, 'forward_impl') and callable(getattr(self, 'forward_impl'))
        self.pert_model = pert_model
        self.channel_broadcaster = ChannelBroadcaster(device, C).to(device)
        self.clipper = None if clipping_net_type is None else get_clipping_network(clipping_net_type)

    def get_in_shape(self, data_shape):
        ds = list(data_shape)
        ds[-2] *= 2
        if not self.indiv_channels:
            ds[-3] = 1
        return ds

    def forward(self, x, c_x=None):
        ndim = x.ndim
        if ndim < 4:
            x = x.unsqueeze(0)

        if not self.indiv_channels and x.shape[-3] != self.C:  # hacky! for segment
            x = self.channel_broadcaster(x)

        x1 = self.forward_impl(x)

        if self.resolution_multiple > 1:
            r, i = x[..., :x.shape[-2]//2, :], x[..., x.shape[-2]//2:, :]
            r2, i2 = torch.flip(r, [0, 1]), -torch.flip(i, [0, 1])
            IW_1N_r, IW_1N_i = torch.real(self.IW_1N).to(torch.float), torch.imag(self.IW_1N).to(torch.float)
            IW_N1_r, IW_N1_i = torch.real(self.IW_N1).to(torch.float), torch.imag(self.IW_N1).to(torch.float)

            x3_r = (IW_1N_r*r2 - IW_1N_i*i2)
            x3_i = (IW_1N_r*i2 + IW_1N_i*r2)
            x4 = x3_r*IW_N1_r.t() - x3_i*IW_N1_i.t()

            x = x1 + x4
        else:
            x = x1

        if c_x is not None:
            if self.pert_model == "additive":
                x = c_x + x
            elif self.pert_model == "spatial_mult":
                x = c_x * (1 + x)

        if self.clipper is not None:
            x = self.clipper(x)

        if ndim < 4:
            x = x.squeeze(0)
        return x

    def init_DFT(self, N, IW_r, IW_i):
        W1_r = torch.empty([N, 2*N])
        W1_r[:, :N], W1_r[:, N:] = IW_r, -IW_i
        W1_i = torch.empty([N, 2*N])
        W1_i[:, :N], W1_i[:, N:] = IW_i, IW_r
        self.W1_r, self.W1_i = W1_r, W1_i

    @staticmethod
    def real_z(z):
        assert z.shape[-1] % 2 == 1, f"{z.shape}, but need input dimension to be odd"

        r1 = z[..., :z.shape[-2]//2, :]
        r21 = torch.flip(r1[..., :1, 1:], [z.ndim-1])
        r22 = torch.flip(r1[..., 1:, 1:], [z.ndim-2, z.ndim-1])

        i1 = z[..., z.shape[-2]//2:, :]
        i21 = -torch.flip(i1[..., :1, 1:], [z.ndim-1])
        i22 = -torch.flip(i1[..., 1:, 1:], [z.ndim-2, z.ndim-1])

        z_ = torch.cat([r21, r22, i21, i22], dim=z.ndim-2)
        z_hat = torch.cat([z, z_], dim=z.ndim-1)
        return z_hat

    def init_concat(self, N, IW_r, IW_i):
        self.forward_impl = self.forward_concat
        self.init_DFT(N, IW_r, IW_i)
        self.L_W1_r = nn.Linear(2*N, N, bias=bias, dtype=torch.float)
        self.L_W1_i = nn.Linear(2*N, N, bias=bias, dtype=torch.float)
        with torch.no_grad():
            self.L_W1_r.weight.copy_(self.W1_r)
            self.L_W1_i.weight.copy_(self.W1_i)
        self.L_W2_r = nn.Linear(2*N, N, bias=bias, dtype=torch.float)
        with torch.no_grad():
            self.L_W2_r.weight.copy_(self.W1_r)

    def forward_concat(self, z):
        if self.real_signal:
            z_ = z[..., :z.shape[-1]//2+1]
            z = self.real_z(z_)
        x = z.transpose(-2, -1)
        x_r = self.L_W1_r(x)
        x_i = self.L_W1_i(x)
        x2 = torch.cat((x_r, x_i), dim=2).transpose(-2, -1)
        x3 = self.L_W2_r(x2)
        return x3

    def init_non_concat(self, N, IW_r, IW_i):
        self.forward_impl = self.forward_non_concat_mm_lin
        self.init_DFT(N, IW_r, IW_i)
        self.P_W1_r = nn.Parameter(self.W1_r.unsqueeze(0).unsqueeze(0))
        self.P_W1_i = nn.Parameter(self.W1_i.unsqueeze(0).unsqueeze(0))
        self.L_W2_r_lin = nn.Linear(N, N, bias=bias, dtype=torch.float)
        self.L_W2_i_lin = nn.Linear(N, N, bias=bias, dtype=torch.float)
        with torch.no_grad():
            self.L_W2_r_lin.weight.copy_(IW_r)
            self.L_W2_i_lin.weight.copy_(-IW_i)

    def forward_non_concat_mm_lin(self, z):
        if self.real_signal:
            z_ = z[..., :z.shape[-1]//2+1]
            z = self.real_z(z_)
        x_r = torch.matmul(self.P_W1_r, z)
        x_i = torch.matmul(self.P_W1_i, z)
        x1 = self.L_W2_r_lin(x_r)
        x2 = self.L_W2_i_lin(x_i)
        x = x1 + x2
        return x


class IDCT(nn.Module):
    def __init__(self, N, C=3, indiv_channels=False, clipping_net_type="default", **kwargs):
        super().__init__()
        Dinv = torch.FloatTensor(idct(np.eye(N), axis=0, norm='ortho'))
        self.L_Dinv = nn.Linear(N, N, bias=bias, dtype=torch.float)
        with torch.no_grad():
            self.L_Dinv.weight.copy_(Dinv)
        self.indiv_channels = indiv_channels
        self.clipper = None if clipping_net_type is None else get_clipping_network(clipping_net_type)

    def get_in_shape(self, data_shape):
        ds = list(data_shape)
        if not self.indiv_channels:
            ds[-3] = 1
        return ds

    def forward(self, x):
        x1 = self.L_Dinv(x)
        x2 = x1.transpose(-2, -1)
        x3 = self.L_Dinv(x2)
        x4 = x3.transpose(-2, -1)
        if self.clipper is not None:
            x4 = self.clipper(x4)
        return x4


def get_clipping_network(net_type: str = None):
    # from data.augmentation_bp_utils import AddConstant, MulConstant
    # if net_type == "lirpa":
    #     return nn.Sequential(
    #             nn.ReLU(),
    #             MulConstant(-1),
    #             AddConstant(1),
    #             nn.ReLU(),
    #             MulConstant(-1),
    #             AddConstant(1),
    #         )
    # else:
    def clamp01_network(x):
        x1 = 1 + -1*nn.functional.relu(x)
        x2 = 1 + -1*nn.functional.relu(x1)
        return x2
    return clamp01_network


def IDFT_matrix(N):
    # https://www.originlab.com/doc/Origin-Help/InverseFFT2-Algorithm
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp(2j * np.pi / N)
    W = np.power(omega, i * j)
    return W
