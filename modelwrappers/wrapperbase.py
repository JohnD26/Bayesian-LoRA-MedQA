# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from contextlib import suppress
from tqdm import tqdm
import math
import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchmetrics import Accuracy, CalibrationError

from transformers import PreTrainedModel
from peft import PeftModel
from peft.config import PeftConfig

from utils import create_if_not_exists

optimizer_dict = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
}


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
        curr_ind = pred[:, curr_k]
        num_eq = torch.eq(curr_ind, target).sum()
        acc = num_eq / len(output)
        res_total += acc
    return res_total * 100


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WrapperBase(PeftModel):
    """
    Base ModelWrapper for this project.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        """Initializes the model wrapper.

        Args:
            model (PreTrainedModel): The pretrained model to wrap.
            peft_config (PeftConfig): The configuration for parameter-efficient fine-tuning (PEFT).
            args (argparse.Namespace): Arguments with configuration for training.
            accelerator (Accelerator): The accelerator to handle multi-GPU or mixed precision.
            adapter_name (str, optional): The name of the adapter. Defaults to "default".
        """
        super().__init__(model, peft_config, adapter_name)

        self.loss = F.nll_loss
        self.args = args
        self.accelerator = accelerator
        self.target_ids = None

        self.batch_size = args.batch_size
        self.num_epochs = args.n_epochs
        self.num_training_steps = args.max_train_steps
        self.step = 0
        self.num_classes = args.outdim
        self.eval_n_samples = 1

        # Deferral mechanism attributes
        self.deferral_enabled = getattr(args, 'enable_deferral', False)
        self.deferral_threshold = getattr(args, 'deferral_threshold', 0.1)
        self.deferral_metric = getattr(args, 'deferral_metric', 'max_std')
        self.deferral_strategy = getattr(args, 'deferral_strategy', 'exclude')
        self.deferral_log_path = getattr(args, 'deferral_log_path', None)
        self.deferral_log_to_wandb = getattr(args, 'deferral_log_to_wandb', False)
        self.deferral_log_file = None  # Will be initialized during prepare_for_fit_evaluate

        if args.max_train_steps == 0:
            num_training_steps = args.num_samples * args.n_epochs // args.batch_size
        else:
            num_training_steps = args.max_train_steps
        warmup_steps = num_training_steps * args.warmup_ratio
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                # set weight_decay
                "weight_decay": args.opt_wd,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        if args.opt == "adamw" or args.opt == "adam":
            self.opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters,
                lr=args.lr,
                eps=args.adam_epsilon,
                weight_decay=args.opt_wd,
            )
        else:
            self.opt = optimizer_dict[args.opt](
                optimizer_grouped_parameters, lr=args.lr, weight_decay=args.opt_wd
            )
        self.scheduler = get_linear_schedule_with_warmup(
            self.opt, warmup_steps, num_training_steps
        )

    def forward_logits(self, *args, **kwargs) -> torch.Tensor:
        """Defines the forward pass for computing logits. This method is not implemented 
        in the base class and needs to be implemented in subclasses.
        
        Returns:
            torch.Tensor: The logits predicted by the model.
        """
        raise NotImplementedError("Forward not implemented.")

    def fit(self, train_loader, eval_loader):
        """Fits the model using the training data and evaluates it periodically.
        
        Args:
            train_loader (DataLoader): The training data loader.
            eval_loader (DataLoader): The evaluation data loader.
        """
        nll_losses = AverageMeter()
        accs = AverageMeter()
        samples_seen = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            for i, batch in enumerate(train_loader):
                if self.args.dataset_type == "mcdataset":
                    _, golds, _ = batch
                elif self.args.dataset_type == "bertds":
                    golds = batch["labels"]
                else:
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented."
                    )
                logits = self.forward_logits(batch).mean(1)
                output = torch.log_softmax(logits, dim=1)
                nll = self.loss(output, golds, reduction="mean")

                self.accelerator.backward(nll)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                acc = accuracy_topk(output.data, golds)
                acc, nll_loss = acc.item(), nll.detach().cpu().numpy()

                if self.args.dataset_type == "mcdataset":
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[
                            : len(train_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                nll_losses.update(nll_loss, len_batch)
                accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log(
                            {
                                "train_acc": accs.avg,
                                "train_nll_loss": nll_losses.avg,
                                "lr": self.opt.param_groups[0]["lr"],
                            }
                        )

                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    self.evaluate(eval_loader)

    def evaluate(self, eval_loader):
        """Evaluates the model using the evaluation data.

        Args:
            eval_loader (DataLoader): The evaluation data loader.

        Returns:
            tuple: The evaluation results: accuracy, ECE (Expected Calibration Error),
                negative log-likelihood (NLL), and Brier score.
        """
        from utils.deferral_metrics import DeferralMetrics

        self.eval()
        status = self.training
        nlls = AverageMeter()
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric = CalibrationError(**metric_kwargs, n_bins=self.args.num_bins).to(
            self.accelerator.device
        )
        briers = AverageMeter()

        # Initialize deferral metrics if enabled
        deferral_metrics = DeferralMetrics() if self.deferral_enabled else None

        samples_seen = 0
        global_sample_idx = 0  # Track sample index for logging

        for step, batch in enumerate(eval_loader):
            with torch.no_grad() and torch.inference_mode():
                # Unpack batch - handle both 3-tuple (old) and 4-tuple (new) formats
                metadata = None
                if isinstance(batch, tuple) and len(batch) == 4:
                    # New format with metadata
                    batch_data, labels_data, targets_data, metadata_batch = batch
                    if self.args.dataset_type == "mcdataset":
                        batch = (batch_data, labels_data, targets_data)
                    else:
                        batch = batch_data
                    metadata = metadata_batch

                logits = self.forward_logits(
                    batch, sample=True, n_samples=self.eval_n_samples
                ).detach()

                if self.args.dataset_type == "mcdataset":
                    if metadata is None:
                        _, labels, _ = batch
                    else:
                        labels = labels_data
                else:
                    labels = batch["labels"]

                logits, labels = self.accelerator.gather([logits, labels])

                # Also gather metadata if using multi-GPU
                if metadata is not None and self.accelerator.num_processes > 1:
                    # Metadata is a list of dicts, need special handling
                    # For simplicity, only log on main process with local metadata
                    pass

                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        labels = labels[: len(eval_loader.dataset) - samples_seen]
                        logits = logits[: len(eval_loader.dataset) - samples_seen]
                        if metadata is not None:
                            metadata = metadata[: len(eval_loader.dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]

                probs = torch.softmax(logits, dim=-1).mean(dim=1)  # [batch_size, num_classes]

                # Compute per-sample uncertainty if needed
                if self.eval_n_samples > 1:
                    std_batch_avg = torch.softmax(logits, dim=-1).std(dim=1).mean()  # For logging

                    # Compute per-sample uncertainty for deferral
                    if self.deferral_enabled:
                        uncertainty_scores = self._compute_uncertainty(logits, self.deferral_metric)  # [batch_size]
                        prob_std = torch.softmax(logits, dim=-1).std(dim=1)  # [batch_size, num_classes]
                    else:
                        uncertainty_scores = None
                        prob_std = None
                else:
                    std_batch_avg = 0
                    uncertainty_scores = None
                    prob_std = None

                # Get predictions
                predictions = probs.argmax(dim=1)  # [batch_size]

                # Apply deferral logic if enabled
                if self.deferral_enabled and uncertainty_scores is not None:
                    # Determine which samples to defer
                    defer_mask = uncertainty_scores > self.deferral_threshold  # [batch_size]

                    # Process each sample for metrics and logging
                    for i in range(len(labels)):
                        is_deferred = defer_mask[i].item()
                        is_correct = (predictions[i] == labels[i]).item()
                        unc_score = uncertainty_scores[i].item()

                        # Update deferral metrics
                        deferral_metrics.update(is_deferred, is_correct, unc_score)

                        # Log deferred samples (only on main process)
                        if is_deferred and self.accelerator.is_local_main_process and metadata is not None:
                            self._log_deferred_sample(
                                sample_idx=global_sample_idx + i,
                                metadata=metadata[i],
                                predicted_idx=predictions[i].item(),
                                uncertainty_score=unc_score,
                                true_idx=labels[i].item(),
                                is_correct=is_correct,
                                mean_probs=probs[i],
                                std_devs=prob_std[i] if prob_std is not None else None,
                            )

                    # Apply deferral strategy
                    if self.deferral_strategy == "exclude":
                        # Only update metrics for non-deferred samples
                        handle_mask = ~defer_mask
                        if handle_mask.sum() > 0:
                            handled_probs = probs[handle_mask]
                            handled_labels = labels[handle_mask]
                            acc_metric(handled_probs, handled_labels)
                            ece_metric(handled_probs, handled_labels)
                            nll = self.loss(torch.log(handled_probs), handled_labels, reduction="mean")
                            if not torch.isnan(nll):
                                nlls.update(nll)
                            brier = (
                                (handled_probs - F.one_hot(handled_labels, num_classes=logits.size(-1)))
                                .pow(2)
                                .sum(dim=-1)
                                .mean()
                            )
                            briers.update(brier)
                    else:  # "always_predict"
                        # Update metrics for all samples (including deferred)
                        acc_metric(probs, labels)
                        ece_metric(probs, labels)
                        nll = self.loss(torch.log(probs), labels, reduction="mean")
                        if torch.isnan(nll):
                            if self.accelerator.is_local_main_process:
                                print("nll:", nll)
                                print("probs:", probs)
                                print("logits:", logits)
                                exit()
                        nlls.update(nll)
                        brier = (
                            (probs - F.one_hot(labels, num_classes=logits.size(-1)))
                            .pow(2)
                            .sum(dim=-1)
                            .mean()
                        )
                        briers.update(brier)
                else:
                    # No deferral - standard evaluation
                    acc_metric(probs, labels)
                    ece_metric(probs, labels)
                    nll = self.loss(torch.log(probs), labels, reduction="mean")
                    if torch.isnan(nll):
                        if self.accelerator.is_local_main_process:
                            print("nll:", nll)
                            print("probs:", probs)
                            print("logits:", logits)
                            exit()
                    nlls.update(nll)
                    brier = (
                        (probs - F.one_hot(labels, num_classes=logits.size(-1)))
                        .pow(2)
                        .sum(dim=-1)
                        .mean()
                    )
                    briers.update(brier)

                global_sample_idx += len(labels)

        val_acc = acc_metric.compute().item()
        val_ece = ece_metric.compute().item()
        val_nll = nlls.avg
        val_brier = briers.avg
        self.train(status)

        # Compute and log deferral metrics if enabled
        if self.deferral_enabled and deferral_metrics is not None:
            deferral_stats = deferral_metrics.compute()

            if self.accelerator.is_local_main_process:
                # Log to WandB if enabled
                if self.wandb_logger is not None:
                    log_dict = {
                        "val_acc": val_acc,
                        "val_ece": val_ece,
                        "val_nll": val_nll,
                        "std": std_batch_avg,
                        "val_brier": val_brier,
                    }

                    if self.deferral_log_to_wandb:
                        log_dict.update({
                            "val_coverage": deferral_stats["coverage"],
                            "val_deferral_rate": deferral_stats["deferral_rate"],
                            "val_acc_on_handled": deferral_stats["accuracy_on_handled"],
                            "val_acc_if_all_predicted": deferral_stats["accuracy_if_all_predicted"],
                            "val_mean_uncertainty_deferred": deferral_stats["mean_uncertainty_deferred"],
                            "val_mean_uncertainty_handled": deferral_stats["mean_uncertainty_handled"],
                        })

                    self.wandb_logger.log(log_dict)

                # Print deferral summary
                print(f"\n=== Deferral Summary ===")
                print(f"Coverage: {deferral_stats['coverage']:.3f} ({deferral_stats['coverage']*100:.1f}%)")
                print(f"Deferral Rate: {deferral_stats['deferral_rate']:.3f} ({deferral_stats['deferral_rate']*100:.1f}%)")
                print(f"Accuracy on Handled: {deferral_stats['accuracy_on_handled']:.3f}")
                print(f"Accuracy if All Predicted: {deferral_stats['accuracy_if_all_predicted']:.3f}")
                print(f"Mean Uncertainty (Deferred): {deferral_stats['mean_uncertainty_deferred']:.4f}")
                print(f"Mean Uncertainty (Handled): {deferral_stats['mean_uncertainty_handled']:.4f}")
        else:
            if self.accelerator.is_local_main_process:
                if self.wandb_logger is not None:
                    self.wandb_logger.log(
                        {
                            "val_acc": val_acc,
                            "val_ece": val_ece,
                            "val_nll": val_nll,
                            "std": std_batch_avg if self.eval_n_samples > 1 else 0,
                            "val_brier": val_brier,
                        }
                    )

        return val_acc, val_ece, val_nll, val_brier

    def _compute_uncertainty(self, logits: torch.Tensor, metric: str = "max_std") -> torch.Tensor:
        """
        Compute per-sample uncertainty scores.

        Args:
            logits: Tensor of shape [batch_size, n_samples, num_classes]
            metric: Uncertainty metric to use ('max_std', 'mean_std', 'entropy', 'bald')

        Returns:
            Uncertainty scores of shape [batch_size]
        """
        from utils.deferral_metrics import (
            compute_max_std,
            compute_mean_std,
            compute_entropy,
            compute_bald,
        )

        if metric == "max_std":
            return compute_max_std(logits)
        elif metric == "mean_std":
            return compute_mean_std(logits)
        elif metric == "entropy":
            probs = torch.softmax(logits, dim=-1).mean(dim=1)
            return compute_entropy(probs)
        elif metric == "bald":
            return compute_bald(logits)
        else:
            raise ValueError(f"Unknown uncertainty metric: {metric}")

    def _get_deferral_log_path(self) -> str:
        """
        Determine the path for the deferral log file.

        Returns:
            Path to JSONL log file
        """
        if self.deferral_log_path is not None:
            return self.deferral_log_path
        else:
            # Default path: checkpoints/{modelwrapper}/{model}/{dataset}/deferred_samples.jsonl
            log_dir = f"checkpoints/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}"
            create_if_not_exists(log_dir)
            return f"{log_dir}/deferred_samples.jsonl"

    def _log_deferred_sample(
        self,
        sample_idx: int,
        metadata: dict,
        predicted_idx: int,
        uncertainty_score: float,
        true_idx: int,
        is_correct: bool,
        mean_probs: torch.Tensor = None,
        std_devs: torch.Tensor = None,
    ):
        """
        Log a deferred sample to the JSONL file.

        Args:
            sample_idx: Global sample index
            metadata: Dictionary with question, options, answer_idx
            predicted_idx: Predicted answer index (0-3)
            uncertainty_score: Computed uncertainty score
            true_idx: True answer index (0-3)
            is_correct: Whether prediction was correct
            mean_probs: Mean probabilities per class (optional)
            std_devs: Standard deviations per class (optional)
        """
        from utils.deferral_metrics import format_deferred_sample_jsonl

        if self.deferral_log_file is None:
            return

        # Convert tensors to lists if provided
        mean_probs_list = mean_probs.cpu().tolist() if mean_probs is not None else None
        std_devs_list = std_devs.cpu().tolist() if std_devs is not None else None

        jsonl_line = format_deferred_sample_jsonl(
            sample_idx=sample_idx,
            metadata=metadata,
            predicted_answer_idx=predicted_idx,
            uncertainty_score=float(uncertainty_score),
            true_answer_idx=int(true_idx),
            is_correct=bool(is_correct),
            mean_probs=mean_probs_list,
            std_devs=std_devs_list,
            n_samples=self.eval_n_samples,
            dataset=self.args.dataset,
            split="validation",
            uncertainty_metric=self.deferral_metric,
        )

        self.deferral_log_file.write(jsonl_line + "\n")
        self.deferral_log_file.flush()  # Ensure it's written immediately

    def fit_evaluate(self):
        """Performs the fitting and evaluation process, saving the results to checkpoints 
        and logging them to the WandB logger.
        """
        if self.accelerator.is_local_main_process:
            save_folder = f"checkpoints/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}/{self.args.log_path}"
            create_if_not_exists(save_folder)
            logging.basicConfig(
                format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                level=logging.INFO,
                filename=save_folder + "/log.txt",
            )
        with tqdm(
            total=self.args.n_epochs, desc=f"Total Training Epochs", leave=True
        ) as pbar:
            for epoch in range(self.args.n_epochs):
                if self.args.early_stop_steps > 0 and epoch >= self.earlystop_n_epochs:
                    break
                self.args.epoch = epoch
                self.fit(self.train_loader, self.test_loader)
                pbar.update(1)

        if hasattr(self.args, "bayes_eval_n_samples_final"):
            self.eval_n_samples = self.args.bayes_eval_n_samples_final

        val_acc, val_ece, val_nll, val_brier = self.evaluate(self.test_loader)
        logging.info(
            f"val_acc: {val_acc}, val_ece: {val_ece}, val_nll: {val_nll}, val_brier: {val_brier}"
        )
        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "final_val_acc": val_acc,
                        "final_val_ece": val_ece,
                        "final_val_nll": val_nll,
                        "final_val_brier": val_brier,
                    }
                )

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """Prepares the model and data loaders for training and evaluation.
    
        Args:
            dataset (Dataset): The dataset object containing train and test data loaders.
            wandb_logger (optional): The Weights & Biases logger for tracking experiments.
        """
        self.wandb_logger = wandb_logger
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader
        
        if self.args.testing_set == 'train_train_val' or self.args.testing_set == 'train_val_val':
            anchor_loader = dataset.anchor_dataloader
            anchor_loader = self.accelerator.prepare(anchor_loader)
            self.anchor_loader = anchor_loader

        if self.args.dataset_type == "mcdataset":
            self.tokenizer = dataset.tokenizer
            self.target_ids = dataset.target_ids.squeeze(-1)

        num_update_steps_per_epoch = math.ceil(len(train_loader))
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )
        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = math.ceil(
                self.args.early_stop_steps / num_update_steps_per_epoch
            )
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            print("len(train_loader):", len(train_loader))
            print("num of epochs:", self.args.n_epochs)
        self.step = 0

        self.base_model, self.opt, train_loader, test_loader, self.scheduler = (
            self.accelerator.prepare(
                self.base_model, self.opt, train_loader, test_loader, self.scheduler
            )
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

        # Initialize deferral log file if enabled
        if self.deferral_enabled and self.accelerator.is_local_main_process:
            log_path = self._get_deferral_log_path()
            self.deferral_log_file = open(log_path, 'w')
            print(f"Deferral logging enabled. Saving to: {log_path}")
