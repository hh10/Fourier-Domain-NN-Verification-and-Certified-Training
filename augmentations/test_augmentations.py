import torch

import unittest
import os

from .augmentations import InputAugmentor


class AugTests(unittest.TestCase):
    test_dir = "/tmp/FDV_tests/augs"

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.test_dir, exist_ok=True)

    def test_eps(self):
        epoch = 150
        aug_cfg = {
            "spec": "ball",
            "domain": "pixel",
            "pert_model": "additive",
            "pix_eps": 0.03,
            "norm_ord": 2,
            "strategy": "aug",  # aug, adv, robust
            "indiv_channels": True,
            "pert_scheduling": {"profile": "sshaped", "end_epoch": int(0.9*epoch), "start_epoch": int(0.025*epoch)}
        }
        data_info = {"name": "cifar10", "shape": (3, 33, 33), "num_classes": 10, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "data_range": (0, 1)}
        input_processor = InputAugmentor(data_info, aug_cfg)
        x = torch.zeros(2, *data_info["shape"])
        for e in range(epoch):
            input_processor.augment_and_process(e, x)
            print(e, input_processor.pert_eps)


if __name__ == '__main__':
    unittest.main()
