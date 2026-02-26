import os

import math
import sys
import time
from typing import Iterable, Optional

import torch

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA

import wandb

import matplotlib

matplotlib.use("Agg")  # prevents tkinter error
import matplotlib.pyplot as plt

import numpy as np

from timm.data.mixup import Mixup

import util.misc as misc
import util.lr_sched as lr_sched
import util.plot as plot

import numpy as np
import matplotlib.pyplot as plt
import wandb

LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def count_optimizer_params(optimizer):
    return sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.requires_grad
    )


all_echonext_tasks = sorted(
    [
        "lvef_lte_45_flag",
        "lvwt_gte_13_flag",
        "aortic_stenosis_moderate_or_greater_flag",
        "aortic_regurgitation_moderate_or_greater_flag",
        "mitral_regurgitation_moderate_or_greater_flag",
        "tricuspid_regurgitation_moderate_or_greater_flag",
        "pulmonary_regurgitation_moderate_or_greater_flag",
        "rv_systolic_dysfunction_moderate_or_greater_flag",
        "pericardial_effusion_moderate_large_flag",
        "pasp_gte_45_flag",
        "tr_max_gte_32_flag",
        "shd_moderate_or_greater_flag",
    ]
)

selected_echonext_tasks = sorted(
    [
        "lvef_lte_45_flag",
        "aortic_stenosis_moderate_or_greater_flag",
        "mitral_regurgitation_moderate_or_greater_flag",
        "rv_systolic_dysfunction_moderate_or_greater_flag",
        "pasp_gte_45_flag",
    ]
)


def log_12lead_ecg_to_wandb(
    ecg, sample_rate=500, title="12-Lead ECG", tag="ecg/12lead"
):
    """
    ecg: shape (12, T) or (T, 12)
    """

    # Convert torch → numpy if needed
    if hasattr(ecg, "detach"):
        ecg = ecg.detach().cpu().numpy()

    # Ensure shape = (12, T)
    if ecg.shape[0] != 12:
        ecg = ecg.T

    T = ecg.shape[1]
    time = np.arange(T) / sample_rate

    fig, axes = plt.subplots(4, 3, figsize=(15, 8), sharex=True)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(time, ecg[i], linewidth=1)
        ax.set_title(LEAD_NAMES[i])
        ax.grid(True)
        ax.set_ylim(-1.5, 1.5)  # adjust if needed

    fig.suptitle(title, fontsize=16)

    img = wandb.Image(fig)

    plt.close(fig)
    return img


def count_params(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    grad_updates=-1,
    args=None,
):
    model.train(True)
    multilabel = data_loader.dataset.multilabel
    metric_logger = misc.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    training_history = {}

    # required for metrics calculation
    logits, labels = [], []
    if epoch == 0:
        wandb.log({"params_count": count_params(model)})
    for data_iter_step, (samples, targets, targets_mask, pos_embed_y) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        start_time = time.time()

        if epoch == 0 and data_iter_step == 0:
            wandb.log(
                {"ecg/12lead": [log_12lead_ecg_to_wandb(ecg[0]) for ecg in samples[:5]]}
            )

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)

        targets = targets * targets_mask
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)
        if args.downstream_task == "classification" and not multilabel:
            targets_mask = targets_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(samples, pos_embed_y) * targets_mask
            loss = criterion(outputs, targets)
        loss_value = loss.item()

        if not math.isfinite(loss_value) and misc.is_main_process():
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            grad_updates += 1

        total_time = time.time() - start_time

        logits.append(outputs)
        labels.append(targets)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lrs_dict = {
            f'{group["part"]}_lr': group["lr"] for group in optimizer.param_groups
        }
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)

        if args.wandb == True and (data_iter_step % accum_iter) == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if misc.is_main_process():
                wandb.log(
                    {
                        "epoch_1000x": epoch_1000x,
                        "time_per_step[sec]": total_time,
                        "loss": loss_value_reduce,
                        **lrs_dict,
                    },
                    step=epoch_1000x,
                )
        if grad_updates == args.num_grad_updates:
            print(f"{grad_updates} steps reached!")
            break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    wandb.log({"params_count": count_params(model)})
    logits = (
        torch.cat(logits, dim=0).to(device="cpu", dtype=torch.float32).detach()
    )  # (B, num_classes)
    labels = torch.cat(labels, dim=0).to(device="cpu").detach()

    if multilabel:
        probs = torch.nn.functional.sigmoid(logits)
    else:
        probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, )
    training_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    echonext_tasks = (
        selected_echonext_tasks if probs.shape[-1] == 5 else all_echonext_tasks
    )
    if args.downstream_task == "classification":
        if multilabel:
            for ind in range(probs.shape[-1]):
                task_labels = labels[:, ind]
                task_probs = logits[:, ind]
                task_preds = (task_probs >= 0.5).int()
                disease = echonext_tasks[ind]
                f1 = 100 * f1_score(
                    y_true=task_labels, y_pred=task_preds, average="macro"
                )
                precision = 100 * precision_score(
                    y_true=task_labels, y_pred=task_preds, average="macro"
                )
                recall = 100 * recall_score(
                    y_true=task_labels, y_pred=task_preds, average="macro"
                )
                acc = 100 * accuracy_score(y_true=task_labels, y_pred=task_preds)
                acc_balanced = 100 * balanced_accuracy_score(
                    y_true=task_labels, y_pred=task_preds
                )
                auc = 100 * roc_auc_score(
                    y_true=task_labels, y_score=probs[:, 1], average="macro"
                )
                cohen = 100 * cohen_kappa_score(
                    y1=task_labels, y2=logits.argmax(dim=-1)
                )

                training_stats[f"{disease}/f1"] = f1
                training_stats[f"{disease}/precision"] = precision
                training_stats[f"{disease}/recall"] = recall
                training_stats[f"{disease}/acc"] = acc
                training_stats[f"{disease}/acc_balanced"] = acc_balanced
                training_stats[f"{disease}/auroc"] = auc
                training_stats[f"{disease}/cohen"] = cohen

            for metric in [
                "f1",
                "precision",
                "recall",
                "acc",
                "acc_balanced",
                "auroc",
                "cohen",
            ]:
                training_stats[metric] = np.array(
                    [v for k, v in training_stats.items() if k.endswith(metric)]
                ).mean()
        else:
            labels_onehot = torch.nn.functional.one_hot(
                labels, num_classes=-1
            )  # (B, num_classes)
            f1 = 100 * f1_score(
                y_true=labels, y_pred=logits.argmax(dim=-1), average="macro"
            )
            precision = 100 * precision_score(
                y_true=labels, y_pred=logits.argmax(dim=-1), average="macro"
            )
            recall = 100 * recall_score(
                y_true=labels, y_pred=logits.argmax(dim=-1), average="macro"
            )
            acc = 100 * accuracy_score(y_true=labels, y_pred=logits.argmax(dim=-1))
            acc_balanced = 100 * balanced_accuracy_score(
                y_true=labels, y_pred=logits.argmax(dim=-1)
            )
            if args.nb_classes > 2:
                auc = 100 * roc_auc_score(
                    y_true=labels, y_score=probs, average="macro", multi_class="ovr"
                )
            else:
                auc = 100 * roc_auc_score(
                    y_true=labels, y_score=probs[:, 1], average="macro"
                )
            auprc = 100 * average_precision_score(
                y_true=labels_onehot, y_score=probs, average="macro"
            )
            cohen = 100 * cohen_kappa_score(y1=labels, y2=logits.argmax(dim=-1))

            training_stats["f1"] = f1
            training_stats["precision"] = precision
            training_stats["recall"] = recall
            training_stats["acc"] = acc
            training_stats["acc_balanced"] = acc_balanced
            training_stats["auroc"] = auc
            training_stats["auprc"] = auprc
            training_stats["cohen"] = cohen
    elif args.downstream_task == "regression":
        rmse = np.float64(
            root_mean_squared_error(logits, labels, multioutput="raw_values")
        )
        training_stats["rmse"] = rmse if isinstance(rmse, float) else rmse.mean(axis=-1)

        mae = np.float64(mean_absolute_error(logits, labels, multioutput="raw_values"))
        training_stats["mae"] = mae if isinstance(mae, float) else mae.mean(axis=-1)

        pcc = np.concatenate(
            [
                r_regression(logits[:, i].view(-1, 1), labels[:, i])
                for i in range(labels.shape[-1])
            ],
            axis=0,
        )
        training_stats["pcc"] = pcc if isinstance(pcc, float) else pcc.mean(axis=-1)

        r2 = np.stack(
            [r2_score(labels[:, i], logits[:, i]) for i in range(labels.shape[-1])],
            axis=0,
        )
        training_stats["r2"] = pcc if isinstance(pcc, float) else r2.mean(axis=-1)

    # wandb
    if args.wandb == True:
        training_history["epoch"] = epoch
        training_history["grad_updates"] = grad_updates
        if args.downstream_task == "classification":
            for k, v in training_stats.items():
                if k.endswith(
                    (
                        "f1",
                        "precision",
                        "recall",
                        "acc",
                        "acc_balanced",
                        "auroc",
                        "cohen",
                    )
                ):
                    training_history[k] = v
        elif args.downstream_task == "regression":
            training_history["rmse"] = training_stats["rmse"]
            training_history["mae"] = training_stats["mae"]
            training_history["pcc"] = training_stats["pcc"]
            training_history["r2"] = training_stats["r2"]

            if targets.shape[-1] > 1:
                for i in range(targets.shape[-1]):
                    training_history[f"Train/RMSE/{i}"] = rmse[i]
                    training_history[f"Train/MAE/{i}"] = mae[i]
                    training_history[f"Train/PCC/{i}"] = pcc[i]
                    training_history[f"Train/R2/{i}"] = r2[i]

    return training_stats, training_history, grad_updates


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    # switch to evaluation mode
    model.eval()
    multilabel = data_loader.dataset.multilabel
    if args.downstream_task == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.downstream_task == "regression":
        criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    test_history = {}

    # required for metrics calculation
    embeddings, logits, labels = [], [], []

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        images = batch[0]
        images = images.to(device, non_blocking=True)

        target = batch[1]
        target = target.to(device, non_blocking=True)

        target_mask = batch[2]
        target_mask = target_mask.to(device, non_blocking=True)
        target = target * target_mask

        pos_embed_y = batch[3]
        pos_embed_y = pos_embed_y.to(device, non_blocking=True)

        if args.downstream_task == "classification" and not multilabel:
            target_mask = target_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)

        # compute output
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                output = model(images, pos_embed_y)
                output = output * target_mask
                loss = criterion(output, target)
        if data_iter_step == 0:
            wandb.log(
                {"ecg/12lead": [log_12lead_ecg_to_wandb(ecg[0]) for ecg in images[:5]]}
            )

        logits.append(output)
        labels.append(target)

        metric_logger.update(loss=loss.item())

    if args.save_embeddings and misc.is_main_process():
        embeddings = (
            torch.cat(embeddings, dim=0).to(device="cpu", dtype=torch.float32).detach()
        )  # (B, D)
        embeddings_path = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)

        file_name = f"embeddings_test.pt" if args.eval else f"embeddings_{epoch}.pt"
        torch.save(embeddings, os.path.join(embeddings_path, file_name))

    # gather the stats from all processes
    if epoch != -1:
        metric_logger.synchronize_between_processes()

    logits = (
        torch.cat(logits, dim=0).to(device="cpu", dtype=torch.float32).detach()
    )  # (B, num_classes)
    labels = torch.cat(labels, dim=0).to(device="cpu").detach()  # (B, 1)
    if multilabel:
        probs = torch.nn.functional.sigmoid(logits)

    else:
        probs = torch.nn.functional.softmax(logits, dim=-1)
    if args.save_logits and misc.is_main_process():
        logits_path = os.path.join(os.path.dirname(args.resume), "logits")
        if not os.path.exists(logits_path):
            os.makedirs(logits_path)

        file_name = f"logits_test.pt" if args.eval else f"logits_{epoch}.pt"
        torch.save(logits, os.path.join(logits_path, file_name))
        torch.save(
            labels, os.path.join(logits_path, file_name.replace("logits", "labels"))
        )
    echonext_tasks = (
        selected_echonext_tasks if probs.shape[-1] == 5 else all_echonext_tasks
    )
    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.downstream_task == "classification":
        if multilabel:
            for ind in range(probs.shape[-1]):
                task_labels = labels[:, ind]
                task_probs = logits[:, ind]
                task_preds = (task_probs >= 0.5).int()
                disease = echonext_tasks[ind]
                f1 = 100 * f1_score(
                    y_true=task_labels, y_pred=task_preds, average="macro"
                )
                precision = 100 * precision_score(
                    y_true=task_labels, y_pred=task_preds, average="macro"
                )
                recall = 100 * recall_score(
                    y_true=task_labels, y_pred=task_preds, average="macro"
                )
                acc = 100 * accuracy_score(y_true=task_labels, y_pred=task_preds)
                acc_balanced = 100 * balanced_accuracy_score(
                    y_true=task_labels, y_pred=task_preds
                )
                auc = 100 * roc_auc_score(
                    y_true=task_labels, y_score=probs[:, 1], average="macro"
                )
                cohen = 100 * cohen_kappa_score(
                    y1=task_labels, y2=logits.argmax(dim=-1)
                )

                test_stats[f"{disease}/f1"] = f1
                test_stats[f"{disease}/precision"] = precision
                test_stats[f"{disease}/recall"] = recall
                test_stats[f"{disease}/acc"] = acc
                test_stats[f"{disease}/acc_balanced"] = acc_balanced
                test_stats[f"{disease}/auroc"] = auc
                test_stats[f"{disease}/cohen"] = cohen
            for metric in [
                "f1",
                "precision",
                "recall",
                "acc",
                "acc_balanced",
                "auroc",
                "cohen",
            ]:
                test_stats[metric] = np.array(
                    [v for k, v in test_stats.items() if k.endswith(metric)]
                ).mean()
        else:
            labels_onehot = torch.nn.functional.one_hot(
                labels, num_classes=-1
            )  # (B, num_classes)
            f1 = 100 * f1_score(
                y_true=labels, y_pred=logits.argmax(dim=-1), average="macro"
            )
            precision = 100 * precision_score(
                y_true=labels, y_pred=logits.argmax(dim=-1), average="macro"
            )
            recall = 100 * recall_score(
                y_true=labels, y_pred=logits.argmax(dim=-1), average="macro"
            )
            acc = 100 * accuracy_score(y_true=labels, y_pred=logits.argmax(dim=-1))
            acc_balanced = 100 * balanced_accuracy_score(
                y_true=labels, y_pred=logits.argmax(dim=-1)
            )
            if args.nb_classes > 2:
                if len(torch.unique(labels)) > 2:
                    # in case there is only one class in the batch
                    auc = 100 * roc_auc_score(
                        y_true=labels, y_score=probs, average="macro", multi_class="ovr"
                    )
                else:
                    auc = torch.nan
            else:
                print(labels.shape, probs.shape)
                auc = 100 * roc_auc_score(
                    y_true=labels, y_score=probs[:, 1], average="macro"
                )
            if len(torch.unique(labels)) > 2:
                # in case there is only one class in the batch
                auprc = 100 * average_precision_score(
                    y_true=labels_onehot, y_score=probs, average="macro"
                )
            else:
                auprc = torch.nan

            cohen = 100 * cohen_kappa_score(y1=labels, y2=logits.argmax(dim=-1))

            test_stats["f1"] = f1
            test_stats["precision"] = precision
            test_stats["recall"] = recall
            test_stats["acc"] = acc
            test_stats["acc_balanced"] = acc_balanced
            test_stats["auroc"] = auc
            test_stats["auprc"] = auprc
            test_stats["cohen"] = cohen
    else:
        raise ValueError()

    print(
        "* Acc@1 {top1_acc:.2f} Acc@1 (balanced) {acc_balanced:.2f} Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f} AUROC {auroc:.2f} loss {losses:.3f}".format(
            top1_acc=acc,
            acc_balanced=acc_balanced,
            precision=precision,
            recall=recall,
            f1=f1,
            auroc=auc,
            losses=test_stats["loss"],
        )
    )

    # tensorboard
    if log_writer is not None:
        log_writer.add_scalar("perf/test_f1", f1, epoch)
        log_writer.add_scalar("perf/test_precision", precision, epoch)
        log_writer.add_scalar("perf/test_recall", recall, epoch)
        log_writer.add_scalar("perf/test_acc", acc, epoch)
        log_writer.add_scalar("perf/test_acc_balanced", acc_balanced, epoch)
        log_writer.add_scalar("perf/test_auroc", auc, epoch)
        log_writer.add_scalar("perf/test_auprc", auprc, epoch)
        log_writer.add_scalar("perf/test_cohen", cohen, epoch)
        log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

    # wandb
    if args.wandb == True:
        test_history = {"epoch": epoch, "test_loss": test_stats["loss"]}
        if args.downstream_task == "classification":
            for k, v in test_stats.items():
                if k.endswith(
                    (
                        "f1",
                        "precision",
                        "recall",
                        "acc",
                        "acc_balanced",
                        "auroc",
                        "cohen",
                    )
                ):
                    test_history[k] = v

    return test_stats, test_history
