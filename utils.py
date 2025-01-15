import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
import dataclasses
from datetime import datetime
import json
import os
from enum import Enum


def seed_and_configure_everything(seed: int = 0, benchmark: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(benchmark)
    torch.backends.cudnn.deterministic = benchmark
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.allow_tf32 = not benchmark


def get_optim_and_scheduler(network, config):
    params = network.parameters()

    if not hasattr(optim, config["type"]):
        raise ValueError(f"Optimizer {config['type']} not implemented")
    pruned_config = {k: v for k, v in config.items() if k not in ["type", "scheduler"]}
    optimizer = getattr(optim, config["type"])(params, **pruned_config)

    scheduler_cfg = config["scheduler"]
    if not hasattr(optim.lr_scheduler, scheduler_cfg["type"]):
        raise ValueError(f"Optimizer scheduler {scheduler_cfg['type']} not implemented")
    pruned_config = {k: v for k, v in scheduler_cfg.items() if k not in ["type"]}
    scheduler = getattr(optim.lr_scheduler, scheduler_cfg["type"])(optimizer, **pruned_config)
    return optimizer, scheduler


def get_bound_loss(lb):
    lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
    fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
    return torch.nn.functional.cross_entropy(-lb_padded, fake_labels)


def load_model(path, device, model):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        print(f"Loaded model from {path} with recorded")
        try:
            for k, v in ckpt.items():
                if k not in ["state_dict"]:
                    print(f"ckpt also has: {k}")
            model.load_state_dict(ckpt["state_dict"])
        except:
            state_dict = {}
            for k, v in ckpt.items():
                state_dict[k.replace("blocks.", "")] = v
            model.load_state_dict(state_dict)
    else:
        print(f"Loaded CKPT from {path}")
        model = ckpt


def write_vnnlib(x_lb, x_ub, x, y, num_classes, vnnlib_path):
    with open(vnnlib_path, "w") as f:
        nxe = 0
        # dft elems
        for xe_lb, xe_ub in zip(x_lb, x_ub):
            f.write(f"(declare-const X_{nxe} Real)\n")
            nxe += 1
        # x_c elems
        for xe in x:
            f.write(f"(declare-const X_{nxe} Real)\n")
            nxe += 1

        # y elems
        for ni in range(num_classes):
            f.write(f"(declare-const Y_{ni} Real)\n")

        nxe = 0
        for xe_lb, xe_ub in zip(x_lb, x_ub):
            f.write(f"(assert (<= X_{nxe} {xe_ub}))\n")
            f.write(f"(assert (>= X_{nxe} {xe_lb}))\n\n")
            nxe += 1
        for xe in x:
            f.write(f"(assert (<= X_{nxe} {xe}))\n")
            f.write(f"(assert (>= X_{nxe} {xe}))\n\n")
            nxe += 1

        ceg_conds = "\n".join([f"(and (>= Y_{ni} Y_{y}))" for ni in range(num_classes) if ni != y])
        f.write(f"(assert (or \n{ceg_conds}\n))")


def write_vnnlibs(it, x_batch, x_aug_batch, y_batch, num_classes, dirpath):
    x_lb_batch, x_ub_batch = x_aug_batch
    for bi, (x, x_lb, x_ub, y) in enumerate(zip(x_batch, x_lb_batch, x_ub_batch, y_batch)):
        vnnlib_path = os.path.join(dirpath, f"{it}_{bi}.vnnlib")
        x_lb, x_ub, x = x_lb.reshape(-1), x_ub.reshape(-1), x.reshape(-1)
        assert torch.min(x_ub-x_lb) >= 0 and torch.max(x_ub-x_lb) > 0, f"{torch.min(x_ub-x_lb).item()}"
        write_vnnlib(x_lb, x_ub, x, y.item(), num_classes, vnnlib_path)
        # already ensure that the batchsize to this function is 1
        break


@dataclasses.dataclass
class Scheduler:
    profile: str
    end_epoch: int
    start_epoch: int = 0
    min_val: float = 0
    max_val: float = 1
    beta: float = 3

    def __call__(self, epoch: int):
        if epoch <= self.start_epoch:
            return self.min_val
        if epoch >= self.end_epoch:
            return self.max_val

        epoch -= self.start_epoch
        warmup_dur = self.end_epoch - self.start_epoch
        if self.profile == "linear":
            return self.min_val + (self.max_val - self.min_val) * epoch / warmup_dur
        elif self.profile == "exp":
            return self.min_val + (self.max_val - self.min_val) / 2 ** (warmup_dur - epoch)
        elif self.profile == "sshaped":
            e = epoch / warmup_dur
            return self.min_val + (self.max_val - self.min_val) * (e ** self.beta / (e ** self.beta + (1 - e) ** self.beta))
        elif self.profile == "sigmoid":
            phase = 1.0 - epoch / warmup_dur
            return float(np.exp(-5.0 * phase * phase))
        raise NotImplementedError


class LogFrequency(Enum):
    PER_CKPT = 1
    PER_UPDATE = 2


class StatsKeeper(object):
    def __init__(self, name, stats_computer=None, tb=None, log_freq=LogFrequency.PER_CKPT):
        self.name = name
        self.stats_computer = stats_computer
        self.scalar = False
        self.checkpoints = []
        self.tb, self.tf_update_cnt = tb, 0
        self.log_freq = log_freq

    def update(self, *args, **kwargs):
        value = self.stats_computer(*args, **kwargs)
        self.value, self.scalar = self.scalar_value(value)
        if self.tb is not None and self.scalar and self.log_freq == LogFrequency.PER_UPDATE:
            self.tb.add_scalar(self.name, self.value, self.tf_update_cnt)
            self.tf_update_cnt += 1

    def checkpoint(self, value=None, then_reset=True, print_ckpt=False):
        # self.update(value)
        if value is None:
            assert self.stats_computer is not None
            value = self.stats_computer.compute()
        self.value, self.scalar = self.scalar_value(value)
        self.checkpoints.append(self.value)
        ckpt_summary = f"{self.name}: {self.value}"
        if print_ckpt: print(ckpt_summary)
        if self.tb is not None and self.scalar and self.log_freq == LogFrequency.PER_CKPT:
            self.tb.add_scalar(self.name, self.value, len(self.checkpoints))
        if then_reset and self.stats_computer is not None:
            self.stats_computer.reset()
        return ckpt_summary

    @staticmethod
    def scalar_value(x):
        is_scalar = type(x) in [float, int, torch.tensor] or x.ndim < 2
        value = x.item() if (type(x) in [torch.tensor, torch.Tensor] and torch.numel(x) == 1) else x.cpu().numpy() if type(x) in [torch.tensor, torch.Tensor] else x
        return value, is_scalar

    @staticmethod
    def get_stats(stats, name: str = None, ckpts=False, as_str=True, only_scalar=True):
        stats = {k: v for k, v in stats.items() if (not only_scalar or v.scalar) and (name is None or name.lower() in k.lower())}
        val_fmt = lambda val: f"{val:.4f}" if type(val) in [float, int] else (f"{val.item():.4f}" if type(val) == torch.tensor else f"{val}")
        stats_dict = {"{:<5}".format(k): val_fmt(v.checkpoints[-1] if ckpts else v.value) for k, v in stats.items()}
        if as_str:
            return stats, ", ".join([f"{k}: {v}" for k, v, in stats_dict.items()]), ", ".join([f"{k[0]}:{v}" for k, v, in stats_dict.items()])
        return stats, stats_dict, list(stats_dict.values())


class AverageCalc(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum, self.count = 0., 0.

    def compute(self):
        return self.sum / max(self.count, 1)

    def __call__(self, value, count=1):
        self.sum += value
        self.count += max(count if isinstance(count, int) else len(count), 1)
        return value / max(count, 1)


class Logger():
    def __init__(self, args):
        self.args = args
        self.log_dir = self.get_results_dirname(args)
        self.tb = None
        self.handlers = {}
        self.time_last_log = datetime.now()
        self.last_epoch = None

    def init_tensorboard(self):
        self.tb = SummaryWriter(self.log_dir)

    def save_config(self, config):
        with open(os.path.join(self.log_dir, 'config.json'), 'w', encoding='utf-8') as file:
            run_data = {"config": config, "args": vars(self.args)}
            json.dump(run_data, file, indent=4)
        self.last_epoch = config["training"]["epoch"]+1

    @staticmethod
    def get_results_dirname(args):
        if args.action == "train":
            assert args.config is not None
            meta_name = args.config.replace(".py", "").replace("/", "_")
            folder_name = os.path.join("/tmp/FDV_exps" if args.dry_run else "results", meta_name)
        else:
            assert args.model_path is not None
            folder_name = os.path.join(os.path.dirname(args.model_path), "evaluations")
        time_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = os.path.join(folder_name, time_name)
        os.makedirs(name)
        return name

    def open(self, fname: str):
        path = os.path.join(self.log_dir, fname)
        if path not in self.handlers:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.handlers[path] = open(path, 'a+', buffering=1)
        return fname

    def close(self):
        for path in list(self.handlers.keys()):
            handler = self.handlers.pop(path)
            handler.close()

    def printf(self, string: str, fname: str):
        path = os.path.join(self.log_dir, fname)
        if path in self.handlers:
            self.handlers[path].write(string)
        else:
            with open(path, 'a+') as f:
                f.write(string)

    def log_values(self, stats, phase, pbar=None, pbar_stat_keys=None, fname=None, epoch=None):
        assert fname is not None or pbar is not None
        if fname is not None:
            header = f"{phase}, {epoch}" if epoch is not None else phase
            stats_str = StatsKeeper.get_stats(stats)[1]
            self.printf(f"{header}, {stats_str}\n", fname)
        if pbar is not None:
            strs = [StatsKeeper.get_stats(stats, k)[2] for k in pbar_stat_keys]
            pbar.set_description(", ".join(strs))

    def log_checkpoints(self, stats, phase, fname='results.txt', epoch=None, print_terminal=False):
        header = f"{phase}, {epoch}" if epoch is not None else phase
        time_now = datetime.now()
        secs_taken = (time_now - self.time_last_log).total_seconds()
        self.time_last_log = time_now
        header += f", time: {secs_taken}"
        stats_str = StatsKeeper.get_stats(stats, only_scalar=epoch is not self.last_epoch)[1]
        self.printf(f"{header}, {stats_str}\n", fname)
        if print_terminal: print(f"{header}, {stats_str}")
