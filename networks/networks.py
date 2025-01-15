# no normalization within the models due to focus of experiments being on input domains
import torch
from torch import nn

import numpy as np

from . import ResNets
from .network_impl import ConvFCNet, Normalization, SegmentInterpolator, MultiInputSequential
from .manual_init import manual_init


class AugmentedNetwork(nn.Module):
    def __init__(self, model, dataset_info, normalize: bool, aug_cfg={}, initialize: bool = False):
        super().__init__()
        self.model, self.device, self.norm_net, self.domain_aug_net = model, model.device, None, None
        self.in_shape, aug_domain_shape, data_shape, self.out_shape = None, dataset_info["shape"], dataset_info["shape"], dataset_info["num_classes"]
        self.aug_cfg = aug_cfg or {}
        self.aug_domain = self.aug_cfg.get("domain", "pixel")

        networks = []  # used for bounding the prepending network

        # prepend the augmentor subnetwork
        if self.aug_cfg.get("spec", "ball") == "segment":
            self.in_shape = (1,)

        # prepending specification and augmentation layers
        if self.aug_domain != "pixel" and self.aug_cfg.get("network_decoding", True):
            if self.aug_domain == "fourier":
                from augmentations import IDFT
                self.domain_aug_net = IDFT(data_shape[-1], C=data_shape[-3], real_signal=False,
                                           pert_model=self.aug_cfg.get("pert_model", "additive"),
                                           indiv_channels=self.aug_cfg.get("indiv_channels", True),
                                           device=self.device).to(self.device)
            elif self.aug_domain == "cosine":
                from augmentations import IDCT
                self.domain_aug_net = IDCT(data_shape[-1], C=data_shape[-3], indiv_channels=self.aug_cfg.get("indiv_channels", True)).to(self.device)
            else:
                raise NotImplementedError(f"Decoding network for {self.domain_aug_net} not supported")
            aug_domain_shape = self.domain_aug_net.get_in_shape(data_shape)
            self.in_shape = self.in_shape or aug_domain_shape
            networks += [self.domain_aug_net]

        # normalization layers
        if normalize:
            assert dataset_info is not None, "Dataset info (nchannels, mean, std) needed to normalize"
            self.norm_net = Normalization(self.device, data_shape[0], dataset_info["mean"], dataset_info["std"])
            networks += [self.norm_net]

        self.in_shape = self.in_shape or data_shape
        self.subnetworks = networks + [model]
        self.bounded_model = None

        # prep for bounded module
        if self.aug_cfg.get("strategy", None) == "robust":
            if self.aug_cfg.get("spec", "ball") == "segment":
                xs = (torch.zeros((4, *aug_domain_shape)), torch.ones((4, *aug_domain_shape)),)
            else:
                xs = (torch.zeros((4, *data_shape)),)
            self.init_bounded_subnet(initialize, *xs)

    def forward_encoding_network(self, x=None, c_x=None, x1=None, alpha=None):
        x_dec = x
        if x1 is not None:
            # for segment queries:
            # the augmentation/interpolation is basically btw x, x1 with alpha \in [0, pix_eps]
            # the adversary, the alpha is provided
            if not torch.is_tensor(alpha):
                alpha = torch.ones((x.shape[0], 1, 1, 1))*alpha if alpha else torch.rand((x.shape[0], 1, 1, 1)) * self.aug_cfg.get("pix_eps", 1)
            alpha = alpha.float().to(self.device)
            x_dec = SegmentInterpolator(self.device, self.aug_cfg.get("domain", "pixel"), x, x1)(alpha)

        if self.domain_aug_net:
            domain_ins = [x_dec, c_x] if self.aug_domain == "fourier" and self.aug_cfg.get("spec", "ball") == "ball" else [x_dec]
            x_dec = self.domain_aug_net(*domain_ins)

        return x_dec

    def forward(self, x=None, x_dec=None, **kwargs):
        if x_dec is None:
            x_dec = self.forward_encoding_network(x, **kwargs)

        x_norm = self.norm_net.to(self.device)(x_dec) if self.norm_net else x_dec
        return self.model(x_norm), x_dec, x_norm

    def get_augmented_network(self, x0, x1=None):
        networks = self.subnetworks
        if x1 is not None:
            networks = [SegmentInterpolator(self.device, self.aug_cfg.get("domain", "pixel"), x0, x1), *self.subnetworks]
        return MultiInputSequential(*networks).to(self.device)

    def init_bounded_subnet(self, initialize, x0, x1=None):
        import sys
        sys.path.insert(0, "verifiers")
        from auto_LiRPA import BoundedModule

        x0 = x0.to(self.device)

        if x1 is not None:  # segment queries
            dummy_input, x1 = (torch.zeros((x0.shape[0], 1, 1, 1)).to(self.device),), x1.to(self.device)
        else:  # ball queries
            if self.aug_domain == "fourier":
                dummy_input = (torch.zeros((x0.shape[0], *self.in_shape)).to(self.device), x0)  # fourier ball is always zero-centered
            else:
                dummy_input = (x0,)

        augmented_model = self.get_augmented_network(x0, x1)
        bounded_model = BoundedModule(augmented_model, dummy_input, device=self.device).to(self.device)
        if initialize:
            manual_init("ibp", augmented_model, bounded_model)
        assert torch.allclose(bounded_model(*dummy_input), augmented_model(*dummy_input)), "Point propagation of augmented and its bounded version should be equal"

        if x1 is not None:
            # if bounding for a segment, will need to reinitialise the model for every segment batch so no point in assigning a self attr, just returning
            return bounded_model, dummy_input
        self.bounded_model = bounded_model

    def forward_bounds(self, x_pt, y, x_bounds=None, x1_pt=None):
        from auto_LiRPA import BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        from auto_LiRPA.utils import get_spec_matrix

        if x1_pt is not None:  # segment queries
            bounded_model, pert_pt = self.init_bounded_subnet(False, x_pt, x1_pt)
            norm_ord, alpha = np.inf, self.aug_cfg.get("pix_eps", 1)
            ptb = PerturbationLpNorm(norm=np.inf, x_L=torch.zeros((x1_pt.shape[0], 1, 1, 1)).to(self.device), x_U=alpha*torch.ones((x1_pt.shape[0], 1, 1, 1)).to(self.device))
            model_ins = (BoundedTensor(pert_pt[0], ptb).to(self.device),)

        else:  # ball queries
            norm_ord, eps = self.aug_cfg.get("norm_ord", np.inf), self.aug_cfg.get("pix_eps", None)
            bounded_model = self.bounded_model
            if self.aug_domain == "fourier":
                norm_ord = np.inf
                ptb = PerturbationLpNorm(norm=norm_ord, x_L=x_bounds[0], x_U=x_bounds[1])
                # zero-centered for fourier, so pert_pt[0] all zeros, pert_pt[1] == x_pt
                pert_pt = (torch.zeros((x_pt.shape[0], *self.in_shape)).to(self.device), x_pt)
                model_ins = (BoundedTensor(pert_pt[0], ptb).to(self.device), pert_pt[1])
            else:
                assert norm_ord in [2, np.inf] and (np.isinf(norm_ord) or eps is not None), f"norm_ord: {norm_ord}, eps: {eps} (must specify for norm_ord other than linf)"
                ptb = PerturbationLpNorm(norm=np.inf, x_L=x_bounds[0], x_U=x_bounds[1])
                pert_pt, model_ins = (x_pt,), (BoundedTensor(x_pt, ptb).to(self.device),)  # absolute for pixels around x_pt

        c = get_spec_matrix(pert_pt[0], y, self.out_shape)
        if self.aug_domain in ["pixel"]:
            bounds_config = {"method": "ibp", "IBP": True}
        else:
            # by default use forward+backward for say evals
            bounds_config = self.aug_cfg.get("bounds_config", {"method": "forward+backward", "IBP": False})

        if bounds_config["method"] == "ibp":
            assert norm_ord == np.inf, f"norm_ord: {norm_ord}, bound lower using C doesn't seem to work correctly with ibp for norm other than np.inf"

        # print(bounds_config)
        y_hat_lb, _ = bounded_model.compute_bounds(x=model_ins, C=c, bound_lower=True, **bounds_config)
        return y_hat_lb


def CNN7(n_class, in_dim, in_ch, bn, bn2):
    return ConvFCNet(n_class, in_dim, in_ch,
                     conv_widths=[4, 4, 8, 8, 8], kernel_sizes=[3, 3, 3, 3, 3],
                     linear_sizes=[512], strides=[1, 1, 2, 1, 1], paddings=[1, 1, 1, 1, 1],
                     net_dim=None, bn=bn, bn2=bn2)


def CifarDeep():
    return nn.Sequential(
            nn.Conv2d(3, 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )


networks_map = {
    'resnet10': ResNets.resnet10,
    'resnet18': ResNets.resnet18,
    'resnet50': ResNets.resnet50,
    'linear': nn.Linear,
    # 'cosine': CosineClassifier,
}


def get_network_(name):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return networks_map[name](**kwargs)
    return get_network_fn


def get_network(net_config, device, data_info):
    def prune_config(d):
        return {k: v for k, v in d.items() if k not in ["name"]}

    if "backbone" in net_config and "head" in net_config:
        backbone = get_network_(net_config["backbone"]["name"])(**prune_config(net_config["backbone"]))
        head = get_network_(net_config["head"]["name"])(**prune_config(net_config["head"]))
        network = nn.Sequential(backbone, head).to(device)
    else:
        net_name = net_config["name"]
        cfg = prune_config(net_config)
        if net_name == "CNN7":
            network = CNN7(data_info["num_classes"], data_info["shape"][-1], data_info["shape"][0], **cfg).to(device)
        elif net_name == "cifar_deep":
            network = CifarDeep().to(device)
        else:
            network = get_network_(net_config["name"])(**cfg).to(device)
    network.device = device
    return network


# ad hoc for the dataset size used in exps; todo: shift hardcodes to class init arguments
class AugmentedONNX(nn.Module):
    def __init__(self, model: AugmentedNetwork):
        super().__init__()
        self.model = model
        self.dft_channels = 1
        self.x_shape = (1, 3, 33, 33)
        self.in_shape = (1, 1, (self.dft_channels*2+3)*33, 33)
        self.dummy_input = torch.randn(self.in_shape)

    def forward(self, zerodftx):
        # 1,1,(nc*2+3)*33, 33, of which nc*2 are idfts and last 3 are x
        ndfts = self.dft_channels*2*33
        zero_dft, x = zerodftx[:, :, :ndfts, :], zerodftx[:, :, ndfts:, :]
        x = x.reshape(self.x_shape)
        return self.model(zero_dft, c_x=x)[0]


if __name__ == '__main__':
    # backbone = {"name": "resnet50"}
    # head = {
    #     "name": "linear",
    #     "in_features": 1120,
    #     "out_features": 12,
    # }
    # networks = {"backbone": backbone, "head": head}
    networks = {"name": "cifar_deep"}
    dataset_info = {"shape": (3, 33, 33), "num_classes": 10}
    x = torch.randn((2, *dataset_info["shape"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_network(networks, device, dataset_info)
    print(model)
    logits = model(x)
    print("x out", logits.shape)

    networks = {"name": "CNN7", "bn": True, "bn2": True}
    model = get_network(networks, "cpu", dataset_info)
    print(model)
    logits = model(x)
    print("x out", logits.shape)
