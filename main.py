import torch

import argparse
import os
import json

from utils import seed_and_configure_everything
from runner import Runner

seed_and_configure_everything(benchmark=False)
    
parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", action='store_true')
parser.add_argument('--action', default="train", choices=["train", "eval"])
parser.add_argument("--config", help="Experiment config", default=None)
parser.add_argument("--cpu", action='store_true')
parser.add_argument("--eps_linf", default=8/255, type=float, choices=[0.01/255, 0.1/255, 1/255, 2/255, 8/255, 9/255, 16/255, 32/255, 48/255, 64/255])
parser.add_argument("--pert_model", default="additive", choices=["additive", "spatial_mult"])
parser.add_argument("--R", type=int, default=2)
# training args
parser.add_argument("--epochs", type=int)
# eval args
parser.add_argument("--model_path", default=None)
parser.add_argument("--cifar10c_severity", type=int, default=5, choices=[1, 2, 3, 4, 5])

args = parser.parse_args()


def import_config_from_file(path):
    module = path.replace(".py", "").replace("/", ".")
    print(f"\nLoading config from {path}\n")
    return __import__(module, fromlist=[""]).config(epochs=args.epochs, eps_linf=args.eps_linf, pert_model=args.pert_model, R=args.R, cifar10c_severity=args.cifar10c_severity)


if args.action == "train":
    assert args.config is not None
    config = import_config_from_file(args.config)
else:
    assert args.model_path is not None
    with open(f"{os.path.dirname(args.model_path)}/config.json") as cf:
        config = json.load(cf)["config"]
    if args.config is not None:
        eval_config = import_config_from_file(args.config)
        for k, v in eval_config.items():
            config[k] = v

device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
runner = Runner(args, config, device)

if args.action == "train":
    runner.do_training()
else:
    runner.do_eval()
