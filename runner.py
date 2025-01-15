import torch
from torch import nn
import torchvision
import torchmetrics

from tqdm import tqdm
import os
from time import time

from data import get_dataset_info, get_dataset, get_data_loaders
from networks import get_network, AugmentedNetwork, AugmentedONNX
from augmentations import InputAugmentor
from utils import load_model, get_optim_and_scheduler, StatsKeeper, AverageCalc, Logger, get_bound_loss, LogFrequency, write_vnnlibs


class Runner:
    def __init__(self, args, config, device):
        self.args, self.config, self.device = args, config, device

        # datasets
        self.data_info = get_dataset_info(**self.config["datasets"]["info"])
        # get target and validation datasets and loaders for evaluation
        self.eval_loaders, normalize_in_net_target, normalize_in_net_val = {}, None, None
        if "target_dataset" in self.config["datasets"]:
            self.target_datasets = get_dataset(False, self.data_info, **self.config["datasets"]["target_dataset"])
            self.eval_loaders = get_data_loaders("test", self.target_datasets, **self.config["datasets"]["target_dataset"])
            normalize_in_net_target = not self.config["datasets"]["target_dataset"].get("normalize", False)
        if "known_dataset" in self.config["datasets"]:
            known_data_val_cfg, val_target_datasets = self.config["datasets"]["known_dataset"].get("val", self.config["datasets"]["known_dataset"]), None
            val_data = get_dataset(False, self.data_info, **known_data_val_cfg)
            if "sample_datasets" in known_data_val_cfg:
                val_target_datasets = {k: self.target_datasets[f"{self.data_info['name']}_{k}"] for k in known_data_val_cfg.pop("sample_datasets")}
            self.eval_loaders["val"] = get_data_loaders("val", val_data, target_datasets=val_target_datasets, **known_data_val_cfg)
            normalize_in_net_val = not self.config["datasets"]["known_dataset"].get("normalize", False)
        assert (normalize_in_net_target == normalize_in_net_val) or (normalize_in_net_target is None ^ normalize_in_net_val is None)
        self.normalize_in_net = normalize_in_net_target or normalize_in_net_val

        # models (incl. network prepends for domain augmentation, spec creation, robustness (training/check))
        self.model = get_network(self.config["networks"], device, self.data_info)
        if self.args.model_path is not None:
            load_model(self.args.model_path, self.device, self.model)
        # self.model = torch.load(self.args.model_path)
        # self.model.device = self.device
        self.model = AugmentedNetwork(self.model, self.data_info, normalize=self.normalize_in_net, aug_cfg=self.config.get("aug", {}), initialize=self.args.action == "train" and self.args.model_path is None)
        print(self.model)

        self.logger = Logger(self.args)
        self.stats = RunnerStats(self.data_info["num_classes"], self.device, [*self.eval_loaders.keys(), *self.config.get("test_augs", {}).keys()])

    def prepare_batch(self, batch, augmentor=None, aug_cfg={}, model=None, epoch=None, subsample_indices=None):
        x, y, x_aug, x1 = batch[0], batch[1], None, batch[3] if aug_cfg.get("spec", None) == "segment" else None
        if subsample_indices is not None:
            x, y = x[subsample_indices].squeeze(1), y[subsample_indices].squeeze(1)
            if x1 is not None:
                x1 = x1[subsample_indices].squeeze(1)

        if augmentor is not None:
            # augmentation output can be an x ([BS, *input_shape]) or x_bounds ([BS, 2, *input_shape]) dep on training_type
            if aug_cfg.get("strategy", None) is None or (epoch is not None and aug_cfg.get("strategy", None) == "adv" and self.best_val_acc < 0.55):
                x, x1 = augmentor.process(x, x1)
            else:
                aug_out = augmentor.augment_and_process(epoch, x, x1=x1, model=model, y=y)
                x, x_aug = aug_out[:2]
                if x1 is not None:
                    x1, x1_aug = aug_out[2:4]
                    # x, x_aug = torch.cat([x, x1]), torch.cat([x_aug, x1_aug])
                    x1 = x1.to(self.device)
                if x_aug is not None:
                    x_aug = x_aug.to(self.device)
            # x_aug: is the augmented or adversarial (around ball or segment) or bounds. Not that for segment, bounds are not used.

        return x.to(self.device), y.to(self.device), x_aug, x1

    def do_eval(self, epoch=None):
        only_evaluation_run = not hasattr(self.stats, "eval")
        if only_evaluation_run:
            self.logger.save_config(self.config)
            self.stats.setup_eval_stats()
        evaluations = [(phase, loader, {}) for phase, loader in self.eval_loaders.items()] + [(aug_type, self.eval_loaders["val"], aug_cfg) for aug_type, aug_cfg in self.config["test_augs"].items()]
        self.model.model.eval()
        with torch.no_grad():
            for (phase, loader, aug_cfg) in evaluations:
                if only_evaluation_run or self.args.dry_run:
                    print(f"Starting {phase} eval")
                    adv_counter = 0
                model = AugmentedNetwork(self.model.model, self.data_info, normalize=self.normalize_in_net, aug_cfg=aug_cfg)
                if only_evaluation_run or self.args.dry_run:
                    print(f"Phase model: {model}")
                    eval_dir = os.path.join(os.path.dirname(self.args.model_path), "evaluations")
                    if "fourier" in phase.lower():
                        # save the augmented NN in onnx format
                        os.makedirs(os.path.join(eval_dir, "onnxs"), exist_ok=True)
                        onnxpath = os.path.join(eval_dir, f"onnxs/{phase}_augmented.onnx")
                        model_onnx = AugmentedONNX(model)
                        torch.onnx.export(model_onnx, model_onnx.dummy_input.to(self.device), onnxpath, verbose=True, export_params=True)

                augmentor = InputAugmentor(self.data_info, aug_cfg) if aug_cfg else None

                pbar = tqdm(loader, leave=False)
                num_vnnlibs, num_robs, total = 0, 0, 0
                for it, batch_data in enumerate(pbar):
                    if (self.args.dry_run and it > 5) or ("val" not in phase.lower() and ((it > 600) or (epoch is not None and it > len(loader)/4))):  # eval upto 1/4th data in training (=>by non-None epoch)
                        break
                    total += 1
                    x_ori, y_ori = batch_data[0].to(self.device), batch_data[1].to(self.device)
                    logits_ori, x_dec, x_dec_norm = self.model(x_dec=x_ori)
                    preds_ori = torch.argmax(logits_ori, dim=1)
                    pred_mask = (preds_ori == y_ori)
                    cor_pred_indices, incor_pred_indices = pred_mask.nonzero(), (~pred_mask).nonzero()
                    if not len(cor_pred_indices):
                        self.stats.eval[f"{phase}ClaAcc"].update(preds_ori, y_ori)
                        if aug_cfg.get("strategy", None) == "robust":
                            self.stats.eval[f"{phase}ClaRobAcc"].update(0, len(y_ori))
                    else:
                        # augment the correctly predicted ones, and check n/w accuracy for their local neighbourhood
                        x, y, x_aug, x1 = self.prepare_batch(batch_data, augmentor, aug_cfg, model, subsample_indices=cor_pred_indices.cpu())
                        if only_evaluation_run and aug_cfg.get("domain", "fourier") and aug_cfg.get("strategy", None) == "robust" and aug_cfg.get("spec", "ball") == "ball":
                            # write vnnlib here from x_aug
                            vnnlibs_dir = os.path.join("verifiers", "vnnlibs", self.data_info['name'], phase)
                            os.makedirs(vnnlibs_dir, exist_ok=True)
                            write_vnnlibs(it, x, x_aug, y, self.data_info["num_classes"], vnnlibs_dir)
                            num_vnnlibs += 1

                        if aug_cfg.get("strategy", None) == "robust":
                            if only_evaluation_run: stt = time()
                            logits_lbounds = model.forward_bounds(x_pt=x, x_bounds=x_aug, x1_pt=x1, y=y)
                            if only_evaluation_run: print(f"computation time: {(time()-stt):.2f}")
                            num_rob = len(logits_lbounds) - torch.sum((logits_lbounds < 0).any(dim=1)).item()
                            num_robs += num_rob
                            # if only_evaluation_run: print(f"robust queries: {num_robs/total} ({num_robs}, {total}, {num_rob}, {y_ori.shape})")
                            self.stats.eval[f"{phase}ClaRobAcc"].update(num_rob, len(y_ori))
                            self.stats.eval[f"{phase}ClaAcc"].update(preds_ori, y_ori)

                        else:
                            if aug_cfg.get("spec", "ball") == "segment":
                                # the augmented x_aug is not used if strategy is aug, as a random alpha is chosen for the segment interpolation & input creation
                                logits_aug = (model(x=x, alpha=x_aug, x1=x1)[0] if aug_cfg.get("strategy", None) == "adv" else model(x=x_aug))[0]
                            else:
                                logits_aug = model(x=x_aug if x_aug is not None else x, c_x=x)[0]  # c_x unused if domain == "pixel"
                            preds_aug = logits_aug.argmax(dim=1)

                            if len(incor_pred_indices) > 0:
                                self.stats.eval[f"{phase}ClaAcc"].update(preds_ori[incor_pred_indices].squeeze(1), y_ori[incor_pred_indices].squeeze(1))
                            self.stats.eval[f"{phase}ClaAcc"].update(preds_aug, y)
                            if False and only_evaluation_run and aug_cfg.get("strategy", None) == "adv":  # and x_aug is not None:
                                # adv_egs_z_cx = [(x_aug[i], x[i]) for i, (pred, y_) in enumerate(zip(preds_aug, y)) if pred != y_]
                                adv_egs_z_cx = [(x_aug if x_aug is not None else x[i], x[i]) for i, (pred, y_) in enumerate(zip(preds_aug, y)) if pred != y_]
                                adv_x_dir = f"{eval_dir}/adv_examples/{phase}"
                                for atype in ["clean", "adv", "combined"]:
                                    os.makedirs(f"{adv_x_dir}/{atype}", exist_ok=True)
                                if len(adv_egs_z_cx) > 0:
                                    for adv_z_, x_ in adv_egs_z_cx:
                                        x_adv_decoded = model.forward_encoding_network(x=adv_z_, c_x=x_)[0]
                                        # adv_x = torch.cat((x_, x_adv_decoded), dim=-1)
                                        # torchvision.utils.save_image(adv_x, f"{adv_x_dir}/{adv_counter}.png")
                                        torchvision.utils.save_image(x_, f"{adv_x_dir}/clean/{adv_counter}.png")
                                        torchvision.utils.save_image(x_adv_decoded, f"{adv_x_dir}/adv/{adv_counter}.png")
                                        adv_counter += 1
                                    # adv_egs_pixel = [torch.cat((x_, model.forward_encoding_network(x=adv_z_, c_x=x_)), dim=-1) for adv_z_, x_ in adv_egs_z_cx]
                                    # if self.logger.tb is None:
                                    #     self.logger.init_tensorboard()
                                    # self.logger.tb.add_image(f'{phase}AdvEgs', torchvision.utils.make_grid(adv_egs_pixel), it)

                    self.logger.log_values(self.stats.eval, phase, pbar=pbar, pbar_stat_keys=['acc'])
                [sc.checkpoint(print_ckpt=only_evaluation_run) for k, sc in self.stats.eval.items() if phase.lower() in k.lower()]
                if only_evaluation_run:
                    self.logger.log_checkpoints(self.stats.eval, "val", epoch=epoch)
        self.logger.log_checkpoints(self.stats.eval, "val", epoch=epoch, print_terminal=only_evaluation_run)

    def _do_epoch(self, epoch):
        self.model.model.train()
        train_pbar = tqdm(self.train_loader, leave=False)
        for it, batch_data in enumerate(train_pbar):
            if self.args.dry_run and it > 2:
                break
            # zero grad and forward
            self.optim.zero_grad()

            x_ori, y_ori = batch_data[0].to(self.device), batch_data[1].to(self.device)
            logits_ori, x_dec, x_dec_norm = self.model(x_dec=x_ori)
            cor_pred_indices = (torch.argmax(logits_ori, dim=1) == y_ori).nonzero().to(self.device)

            x_aug_dec, y, logits_aug, loss_aug = None, [], None, 0

            if "aug" in self.config and len(cor_pred_indices) > 0:
                # augment the correctly predicted ones, and train n/w for accuracy in their local neighbourhood
                x, y, x_aug, x1 = self.prepare_batch(batch_data, self.augmentor, self.config.get("aug", None), self.model, epoch, subsample_indices=cor_pred_indices.cpu())

                if self.config["aug"].get("strategy", None) == "robust":
                    logits_lbounds = self.model.forward_bounds(x_pt=x, x_bounds=x_aug, x1_pt=x1, y=y)  # either x_bounds (ball) or x1_pt (segment) is used
                    loss_aug = get_bound_loss(logits_lbounds)
                    # if self.config["aug"]["bounds_config"]["method"] == "ibp":
                    #     loss_aug *= min(1 - self.augmentor.pert_factor, self.config["aug"].get("minimum_robust_weight", 0.5))

                elif x_aug is not None:
                    if self.config["aug"].get("spec", "ball") == "segment":
                        logits_aug, x_aug_dec, x_aug_dec_norm = self.model(x=x, alpha=x_aug, x1=x1)  # x1 (segment)
                    else:
                        logits_aug, x_aug_dec, x_aug_dec_norm = self.model(x=x_aug, c_x=x)  # c_x (ball)
                    loss_aug = self.criterion(logits_aug, y)

            loss_ori = self.criterion(logits_ori, y_ori)
            total_loss = loss_ori + loss_aug
            # total_loss = loss_ori*len(y_ori) + loss_aug*len(y)
            # total_loss /= (len(y_ori)+len(y))

            # backward and update
            total_loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=8.0)
            self.optim.step()

            # update all batch stats and update the progress bar
            self.stats.update_training_stats(grad_norm, (y_ori, logits_ori, loss_ori), (y, logits_aug, loss_aug) if loss_aug else None)
            self.logger.log_values(self.stats.training, 'train', pbar=train_pbar, pbar_stat_keys=['loss', 'acc'], epoch=epoch, fname=self.log_file)

        # checkpoint the stats (loss and acc) and log results for the epoch
        [sc.checkpoint() for k, sc in self.stats.training.items() if 'loss' in k.lower() or 'acc' in k.lower()]
        self.logger.log_checkpoints(self.stats.training, 'train', epoch=epoch)
        if x_dec is not None:
            nimages2view = 32
            self.logger.tb.add_image('x_norm, x_hat, x_aug_hat',
                torchvision.utils.make_grid(torch.cat([x_dec[:nimages2view], x_aug_dec[:nimages2view], x_aug_dec_norm[:nimages2view]]) if x_aug_dec is not None else  # noqa: E128
                                            torch.cat([x_dec[:nimages2view], x_dec_norm[:nimages2view]])), epoch)

        # eval, log eval_stats and save best val model
        self.do_eval(epoch=epoch)
        if self.stats.eval["valClaAcc"].checkpoints[-1] >= self.best_val_acc:
            self.best_val_acc = self.stats.eval["valClaAcc"].checkpoints[-1]
            save_model(self.logger.log_dir, self.model, {"epoch": epoch, "val_acc": self.best_val_acc})

    def do_training(self):
        # training dataset
        known_data_train_cfg, train_target_datasets = self.config["datasets"]["known_dataset"].get("train", self.config["datasets"]["known_dataset"]), None
        train_data = get_dataset(True, self.data_info, **known_data_train_cfg)
        if "sample_datasets" in known_data_train_cfg:
            train_target_datasets = {k: self.target_datasets[f"{self.data_info['name']}_{k}"] for k in known_data_train_cfg.pop("sample_datasets")}
        self.train_loader = get_data_loaders("train", train_data, target_datasets=train_target_datasets, **known_data_train_cfg)

        # augmentor here doubles as input processor to another domain and optionally sampler (if checking for a segment) and augmenter/adversary/bounds computer
        self.augmentor = None
        if self.config.get("aug", {}):
            self.augmentor = InputAugmentor(self.data_info, self.config["aug"])  # this is essentially for training, test augmentor are created as required in do_eval

        self.logger.save_config(self.config)
        self.log_file = self.logger.open('log.txt')
        self.logger.init_tensorboard()
        self.stats.setup_stats(["Ori"] + (["Aug"] if self.augmentor is not None else []), self.logger.tb)

        # training params (optimizer, scheduler, ...)
        self.optim, self.lr_scheduler = get_optim_and_scheduler(self.model.model if isinstance(self.model, AugmentedNetwork) else self.model,
                                                                self.config["training"]["optimiser"])
        self.criterion = nn.CrossEntropyLoss()

        self.best_val_acc = 0
        epoch_pbar = tqdm(range(1, self.config["training"]["epoch"]+1), desc="Epoch", leave=True)
        for epoch in epoch_pbar:
            if self.args.dry_run and epoch > 2:
                break
            self.stats.training['lr'].checkpoint(self.lr_scheduler.get_last_lr()[0])
            self._do_epoch(epoch)
            if self.augmentor:
                self.stats.training['eps'].checkpoint(self.augmentor.pert_eps)
            self.lr_scheduler.step()
            save_model(self.logger.log_dir, self.model, {"epoch": epoch, "val_acc": self.stats.eval["valClaAcc"].checkpoints[-1], "best_val_acc": self.best_val_acc}, name="latest_model")
            if (epoch > self.config["training"]["epoch"]/10 and epoch % 7 == 0):
                save_model(self.logger.log_dir, self.model, {"epoch": epoch, "val_acc": self.stats.eval["valClaAcc"].checkpoints[-1], "best_val_acc": self.best_val_acc}, name=f"model_epoch{epoch}")
        save_model(self.logger.log_dir, self.model, {"epoch": epoch, "val_acc": self.stats.eval["valClaAcc"].checkpoints[-1], "best_val_acc": self.best_val_acc}, name=f"model_epoch{epoch}")
        self.logger.close()


class RunnerStats:
    def __init__(self, num_classes, device, eval_keys):
        self.device, self.eval_keys = device, eval_keys
        self.metrics_cfg = {"task": "multiclass", "num_classes": num_classes}

    def setup_eval_stats(self, tb=None):
        self.eval = {}
        for phase in self.eval_keys:
            if "rob" in phase.lower():
                self.eval[f"{phase}ClaRobAcc"] = StatsKeeper(f"{phase}ClaRobAcc", AverageCalc(), tb)
            self.eval[f"{phase}ClaAcc"] = StatsKeeper(f"{phase}ClaAcc", torchmetrics.Accuracy(**self.metrics_cfg).to(self.device), tb)
        print("Computing eval stats:", self.eval.keys())

    def setup_stats(self, training_keys, tb):
        self.tb = tb
        self.training = {"lr": StatsKeeper("lr", tb=self.tb), "grad_norm": StatsKeeper("grad_norm", tb=self.tb), "eps": StatsKeeper("eps", tb=self.tb)}
        for k in training_keys:
            self.training[f"Loss{k}"] = StatsKeeper(f"Loss{k}", AverageCalc(), self.tb, log_freq=LogFrequency.PER_UPDATE)
            self.training[f"ClaAcc{k}"] = StatsKeeper(f"ClaAcc{k}", torchmetrics.Accuracy(**self.metrics_cfg).to(self.device), self.tb)
        self.setup_eval_stats(tb)

    def update_training_stats(self, grad_norm, pred_data_ori, pred_data_aug=None):
        self.training["grad_norm"].checkpoint(grad_norm.item())

        def populate_stats(pred_data, label):
            y, logits, loss = pred_data
            self.training[f"Loss{label}"].update(loss.item())
            if logits is not None:
                preds = logits.argmax(dim=1)
                self.training[f"ClaAcc{label}"].update(preds, y)

        populate_stats(pred_data_ori, "Ori")
        if pred_data_aug:
            populate_stats(pred_data_aug, "Aug")


def save_model(dirpath, model, metadata, name="best_model"):
    path = os.path.join(dirpath, f'{name}.pth')
    if isinstance(model, AugmentedNetwork):
        torch.save({**metadata, 'state_dict': model.model.state_dict()}, path)
    else:
        torch.save({**metadata, 'state_dict': model.state_dict()}, path)
