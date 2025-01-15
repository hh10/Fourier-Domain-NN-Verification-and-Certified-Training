import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SegmentInterpolator(nn.Module):
    def __init__(self, device, domain, pt0, pt1):
        super().__init__()
        self.device = device
        assert pt0.shape == pt1.shape, f"shapes: {pt0.shape} {pt1.shape}"

        self.c, self.b = (pt1-pt0)[0], pt0[0]

    def forward(self, alpha):
        return alpha * self.c + self.b


class MultiInputSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def to(self, device):
        for i, _ in enumerate(self):
            self[i] = self[i].to(device)
        return self

    def forward(self, *args):
        x = self[0](*args)
        for module in self[1:]:
            x = module(x)
        return x


class ChannelBroadcaster(nn.Module):
    def __init__(self, device, C):
        super().__init__()
        self.cb = nn.Conv2d(1, C, 1, bias=False).to(device)
        self.cb.weight = nn.Parameter(torch.ones_like(self.cb.weight))

    def forward(self, x):
        assert x.shape[-3] == 1, f"{x.shape}"
        return self.cb(x)


class Normalization(nn.Module):
    def __init__(self, device, num_channels, mean=0., std=1.):
        super().__init__()
        # self.mean_negative = -torch.tensor(mean, dtype=torch.float, device=device).reshape(num_channels, 1, 1)  # torch.nn.Parameter( . , requires_grad=False)
        # self.std_reciprocal = 1/torch.tensor(std, dtype=torch.float, device=device).reshape(num_channels, 1, 1)
        mean_negative = -torch.tensor(mean, dtype=torch.float).reshape(num_channels, 1, 1)  # torch.nn.Parameter( . , requires_grad=False)
        std_reciprocal = 1/torch.tensor(std, dtype=torch.float).reshape(num_channels, 1, 1)
        self.mean_negative = nn.Parameter(mean_negative, requires_grad=False)
        self.std_reciprocal = nn.Parameter(std_reciprocal, requires_grad=False)

        # if num_channels in [3, 4]:
        #     self.mean.data = self.mean.data.view(1, -1, 1, 1)
        #     self.std.data = self.std.data.view(1, -1, 1, 1)
        # elif num_channels in [1, 2]:
        #     self.mean.data = self.mean.data.view(1, -1)
        #     self.std.data = self.std.data.view(1, -1)
        # else:
        #     assert False

    def forward(self, x):
        return (x + self.mean_negative) * self.std_reciprocal


class ConvFCNet(nn.Module):
    def __init__(self, out_size, input_size, input_channel, conv_widths=None,
                 kernel_sizes=[3], linear_sizes=[], depth_conv=None, paddings=[1], strides=[2],
                 dilations=[1], pool=False, net_dim=None, bn=False, bn2=False, max=False, scale_width=True):
        super(ConvFCNet, self).__init__()
        if net_dim is None:
            net_dim = input_size
        if len(conv_widths) != len(kernel_sizes):
            kernel_sizes = len(conv_widths) * [kernel_sizes[0]]
        if len(conv_widths) != len(paddings):
            paddings = len(conv_widths) * [paddings[0]]
        if len(conv_widths) != len(strides):
            strides = len(conv_widths) * [strides[0]]
        if len(conv_widths) != len(dilations):
            dilations = len(conv_widths) * [dilations[0]]

        self.out_size = out_size
        self.input_size = input_size
        self.input_channel = input_channel
        self.conv_widths = conv_widths
        self.kernel_sizes = kernel_sizes
        self.paddings = paddings
        self.strides = strides
        self.dilations = dilations
        self.linear_sizes = linear_sizes
        self.depth_conv = depth_conv
        self.net_dim = net_dim
        self.bn = bn
        self.bn2 = bn2
        self.max = max

        layers = []

        N = net_dim
        n_channels = input_channel
        # dims does what? 
        self.dims = [(n_channels, N, N)]

        for width, kernel_size, padding, stride, dilation in zip(conv_widths, kernel_sizes, paddings, strides, dilations):
            # add all convolutional layers
            if scale_width:
                width *= 16
            N = int(np.floor((N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            layers += [nn.Conv2d(n_channels, int(width), kernel_size, stride=stride, padding=padding, dilation=dilation)]
            if self.bn:
                layers += [nn.BatchNorm2d(int(width))]
            if self.max:
                layers += [nn.MaxPool2d(int(width))]
            layers += [nn.ReLU((int(width), N, N))]
            n_channels = int(width)
            self.dims += 2*[(n_channels, N, N)]

        if depth_conv is not None:
            layers += [nn.Conv2d(n_channels, depth_conv, 1, stride=1, padding=0),
                       nn.ReLU((n_channels, N, N))]
            n_channels = depth_conv
            self.dims += 2*[(n_channels, N, N)]

        if pool:
            layers += [nn.GlobalAvgPool2d()]
            self.dims += 2 * [(n_channels, 1, 1)]
            N = 1

        # flatten in preparation of fully connected layers
        layers += [nn.Flatten()]
        N = n_channels * N ** 2
        self.dims += [(N,)]

        for width in linear_sizes:
            if width == 0:
                continue
            layers += [nn.Linear(int(N), int(width))]
            if self.bn2:
                layers += [nn.BatchNorm1d(int(width))]
            layers += [nn.ReLU(width)]
            N = width
            self.dims += 2*[(N,)]

        layers += [nn.Linear(N, out_size)]
        self.dims += [(out_size,)]

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class CosineClassifier(nn.Module):
    def __init__(
        self,
        num_channels,
        num_classes,
        scale=1.0,
        learn_scale=False,
        bias=False,
        normalize_x=True,
        normalize_w=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize_x = normalize_x
        self.normalize_w = normalize_w

        weight = torch.FloatTensor(num_classes, num_channels).normal_(
            0.0, np.sqrt(2.0 / num_channels)
        )
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        scale_cls = torch.FloatTensor(1).fill_(scale)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=learn_scale)

    def forward(self, x_in):
        assert x_in.dim() == 2
        return cosine_fully_connected_layer(
            x_in,
            self.weight.t(),
            scale=self.scale_cls,
            bias=self.bias,
            normalize_x=self.normalize_x,
            normalize_w=self.normalize_w,
        )

    def extra_repr(self):
        s = "num_channels={}, num_classes={}, scale_cls={} (learnable={})".format(
            self.num_channels,
            self.num_classes,
            self.scale_cls.item(),
            self.scale_cls.requires_grad,
        )
        learnable = self.scale_cls.requires_grad
        s = (
            f"num_channels={self.num_channels}, "
            f"num_classes={self.num_classes}, "
            f"scale_cls={self.scale_cls.item()} (learnable={learnable}), "
            f"normalize_x={self.normalize_x}, normalize_w={self.normalize_w}"
        )

        if self.bias is None:
            s += ", bias=False"
        return s


def cosine_fully_connected_layer(
    x_in, weight, scale=None, bias=None, normalize_x=True, normalize_w=True
):
    # x_in: a 2D tensor with shape [batch_size x num_features_in]
    # weight: a 2D tensor with shape [num_features_in x num_features_out]
    # scale: (optional) a scalar value
    # bias: (optional) a 1D tensor with shape [num_features_out]

    assert x_in.dim() == 2
    assert weight.dim() == 2
    assert x_in.size(1) == weight.size(0)

    if normalize_x:
        x_in = F.normalize(x_in, p=2, dim=1, eps=1e-12)

    if normalize_w:
        weight = F.normalize(weight, p=2, dim=0, eps=1e-12)

    x_out = torch.mm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale.view(1, -1)

    if bias is not None:
        x_out = x_out + bias.view(1, -1)

    return x_out
