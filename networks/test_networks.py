import torch
from torchvision.utils import make_grid, save_image

import numpy as np
import os
import unittest
import itertools
from collections import namedtuple

from .network_impl import SegmentInterpolator
from runner import Runner
from augmentations import InputAugmentor


class NetworkTests(unittest.TestCase):
    test_dir = "/tmp/FDV_tests/networks"
    
    @staticmethod
    def get_runner(spec_type, domain, training_type):
        network_cfg = {"name": "CNN7", "bn": False, "bn2": False}
        dataset_name, test_subdomains = "codan", ["test_night"]
        dataset_cfg = {
            "info": {
                "name": dataset_name,
                "image_size": 33,
            },
            "target_dataset": {
                "dataset_name": dataset_name,
                "subdomains": test_subdomains,
                "batch_size": 4,
                "transform_strategy": "std",
                "num_samples_per_input": 2,
                "sample_from_datasets": "inter",
            },
            "known_dataset": {
                "dataset_name": "codan",
                "batch_size": 1,
                "transform_strategy": "std",
                "num_samples_per_input": 2,
                "sample_datasets": test_subdomains,
            },
        }
        aug_cfg = {
                "spec": spec_type,
                "domain": domain,
                "pert_model": "additive",
                "strategy": training_type,
                "pix_eps": 0.5664020917185318,
                "norm_ord": 2,
                "indiv_channels": False,
                "fourier_mask": {"freq_mask_range": (0, 2)},
                "zero_centered": True,
            }
        cfg = {"networks": network_cfg, "datasets": dataset_cfg, "aug": aug_cfg}
        args = namedtuple('args', ['action', 'model_path', 'config', 'dry_run'])
        runner = Runner(args("train", None, "network_tests", True), cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return runner, aug_cfg

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.test_dir, exist_ok=True)

    def test_networks_for_segment_queries(self):
        specs, domains, trainings = ["segment"], ["fourier", "pixel"], ["aug", "robust"]
        for spec_type, domain, training_type in itertools.product(specs, domains, trainings):
            print(f"\nChecking {spec_type}, {domain}, {training_type}...")
            runner, aug_cfg = self.get_runner(spec_type, domain, training_type)
            aug_cfg["pix_eps"] += 0.75
            print(runner.model)

            val_loader = runner.eval_loaders["val"]
            x, y, _, x1, _, _ = next(iter(val_loader))
            print(x.shape, x1.shape)

            input_processor = InputAugmentor(runner.data_info, aug_cfg)
            x_in, xd_aug, x1_in = input_processor.augment_and_process(None, x, x1)[:3]
            print(x_in.shape, x1_in.shape)

            if training_type == "robust":
                y_hat, y_hat_lb = runner.model(x_pt=x_in, x1_pt=x1_in, x_bounds=xd_aug, y=y)  # x_bounds unused
                self.assertEqual(list(y_hat_lb.shape), [*y_hat.shape[:-1], y_hat.shape[-1]-1])

                aug_net = runner.model.get_augmented_network(x_in, x1_in)
                y_hat_aug = aug_net(torch.zeros((x_in.shape[0], 1, 1, 1)))
                torch.testing.assert_close(y_hat, y_hat_aug)
            else:
                y_hat, x_aug, x_norm = runner.model(x=x_in, x1=x1_in, c_x=x_in)  # c_x unused
                print(f"shapes: x ({x.shape}, {x_aug.shape}, {x_norm.shape}), y_hat ({y_hat.shape})")
                save_image(make_grid(torch.cat([x.to("cpu"), x_aug.to("cpu"), x1.to("cpu")] if x1 is not None else [x.to("cpu"), x_aug.to("cpu")])), f"{NetworkTests.test_dir}/{spec_type}_{domain}_{training_type}_network_proc.png")

    # def test_networks_for_ball_queries(self):
    #     specs, domains, trainings = ["ball"], ["fourier", "pixel"], ["aug", "adv", "robust"]
    #     for spec_type, domain, training_type in itertools.product(specs, domains, trainings):
    #         print(f"\nChecking {spec_type}, {domain}, {training_type}...")
    #         runner, aug_cfg = self.get_runner(spec_type, domain, training_type)
    #         print(runner.model)

    #         val_loader = runner.eval_loaders["val"]
    #         x, y = next(iter(val_loader))[:2]

    #         input_processor = InputAugmentor(runner.data_info, aug_cfg)
    #         x_in, xd_aug = input_processor.augment_and_process(None, x, model=runner.model, y=y)[:2]
    #         print(x.shape, x_in.shape, xd_aug.shape)
    #         x, y, x_in, xd_aug = x.to(runner.device), y.to(runner.device), x_in.to(runner.device), xd_aug.to(runner.device)

    #         if training_type == "robust":
    #             y_hat, y_hat_lb = runner.model(x_pt=x_in, x_bounds=xd_aug, y=y)
    #             self.assertEqual(list(y_hat_lb.shape), [*y_hat.shape[:-1], y_hat.shape[-1]-1])
    #         else:
    #             y_hat, x_aug, x_norm = runner.model(x=xd_aug, c_x=x_in)
    #             print(f"shapes: x ({x.shape}, {x_aug.shape}, {x_norm.shape}), y_hat ({y_hat.shape})")
    #             save_image(make_grid(torch.cat([x, x_aug])), f"{NetworkTests.test_dir}/{spec_type}_{domain}_{training_type}_network_proc_{aug_cfg['pix_eps']}_l{aug_cfg['norm_ord']}.png")

    def test_segment_interpolator(self):
        x, x1 = torch.rand(1, 3, 66, 33), torch.rand(1, 3, 66, 33)
        segment_interpolator = SegmentInterpolator("cpu", "fourier", x, x1)
        x_hat = segment_interpolator(torch.tensor((0.,)).float())
        torch.testing.assert_close(x_hat.unsqueeze(0), x)
        # TODO: complete this after func,
        # x1_hat = segment_interpolator(torch.tensor((1.,)).float())
        # torch.testing.assert_close(x1_hat, x1)


if __name__ == "__main__":
    unittest.main()
    # nt = NetworkTests()
    # nt.setUpClass()
    # nt.test_networks_for_ball_queries()
    # nt.test_networks_for_segment_queries()
