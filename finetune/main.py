import os
import argparse
import json
from typing import Tuple
import numpy as np
import time
from pathlib import Path
import random
import string
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from ast import literal_eval
from typing import List
from baselines import Baseline

import wandb
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup

from util.dataset import TimeSeriesDataset, log_set_distribution
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed_x
from util.callbacks import EarlyStop

import models_vit

from engine_finetune import train_one_epoch, evaluate
from datetime import datetime


def create_optimizer_with_different_lrs(model, head_lr, backbone_lr, wd):

    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    params = []
    if backbone_params != []:
        params.append(
            {"params": backbone_params, "lr": backbone_lr, "part": "backbone"}
        )
    if head_params != []:
        params.append({"params": head_params, "lr": head_lr, "part": "head"})

    optimizer = torch.optim.AdamW(
        params,
        weight_decay=wd,
    )
    return optimizer


def get_args_parser():
    parser = argparse.ArgumentParser("OTiS fine-tuning", add_help=False)

    parser.add_argument(
        "--batch_size",
        default=20,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--num_grad_updates", default=-1, type=int)
    parser.add_argument("--validation_frequency", default=1, type=int)
    parser.add_argument(
        "--accum_iter",
        default=4,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--num_samples", default=-1, type=int, help="Number of samples to use"
    )
    parser.add_argument(
        "--dataset_percentage",
        default=100,
        type=float,
        help="Percentage of train dataset to use",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_baseDeep_patchX",
        type=str,
        metavar="MODEL",
        help="Name of model to train (default: vit_baseDeep_patchX)",
    )
    parser.add_argument(
        "--multilabel",
        action="store_true",
        default=False,
        help="Wheter the task is multilabel",
    )

    parser.add_argument(
        "--univariate",
        action="store_true",
        default=False,
        help="Univariate time series analysis (i.e. treat each variate independently)",
    )

    parser.add_argument(
        "--input_channels", type=int, default=1, metavar="N", help="input channels"
    )
    parser.add_argument(
        "--input_variates", type=int, default=12, metavar="N", help="input variates"
    )
    parser.add_argument(
        "--time_steps", type=int, default=5000, metavar="N", help="input length"
    )
    parser.add_argument(
        "--input_size", default=(1, 12, 5000), type=Tuple, help="samples input size"
    )

    parser.add_argument(
        "--patch_height", type=int, default=1, metavar="N", help="patch height"
    )
    parser.add_argument(
        "--patch_width", type=int, default=24, metavar="N", help="patch width"
    )
    parser.add_argument(
        "--patch_size",
        default=(-1, -1),
        type=Tuple,
        help="patch size - set dinamycally",
    )

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--lin_probe",
        action="store_true",
        default=False,
        help="Wheter to do linear probing",
    )

    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="Wheter to freeze the backbone - but not lin probing",
    )

    parser.add_argument(
        "--unfreeze",
        type=str,
        default="[]",
        help="Pairs of epochs and blocks to unfreeze",
    )

    # Augmentation parameters
    parser.add_argument(
        "--masking_blockwise",
        action="store_true",
        default=False,
        help="Masking blockwise in channel and time dimension (instead of random masking)",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.0,
        type=float,
        help="Masking ratio (percentage of removed patches)",
    )
    parser.add_argument(
        "--mask_c_ratio",
        default=0.0,
        type=float,
        help="Masking ratio in channel dimension (percentage of removed patches)",
    )
    parser.add_argument(
        "--mask_t_ratio",
        default=0.0,
        type=float,
        help="Masking ratio in time dimension (percentage of removed patches)",
    )
    parser.add_argument(
        "--augmentation", default="default", type=str, help="Augmentation to do"
    )
    parser.add_argument(
        "--crop_lower_bnd",
        default=0.75,
        type=float,
        help="Lower boundary of the cropping ratio (default: 0.75)",
    )
    parser.add_argument(
        "--crop_upper_bnd",
        default=1.0,
        type=float,
        help="Upper boundary of the cropping ratio (default: 1.0)",
    )

    parser.add_argument(
        "--jitter_sigma",
        default=0.2,
        type=float,
        help="Jitter sigma N(0, sigma) (default: 0.2)",
    )
    parser.add_argument(
        "--rescaling_sigma",
        default=0.5,
        type=float,
        help="Rescaling sigma N(0, sigma) (default: 0.5)",
    )
    parser.add_argument(
        "--ft_surr_phase_noise",
        default=0.075,
        type=float,
        help="Phase noise magnitude (default: 0.075)",
    )
    parser.add_argument(
        "--freq_shift_delta",
        default=0.005,
        type=float,
        help="Delta for the frequency shift (default: 0.005)",
    )

    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        required=True,
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=None,
        required=True,
        help="backbone learning rate (used if we unfreeze backbone)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 4",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Callback parameters
    parser.add_argument(
        "--patience",
        default=-1,
        type=float,
        help="Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)",
    )
    parser.add_argument(
        "--max_delta",
        default=0.1,
        type=float,
        help="Early stopping threshold (val has to be worse than (train+delta)) (default: 0)",
    )

    # Criterion parameters
    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        default=False,
        help="Apply inverse frequency weighted loss (default: False)",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument(
        "--finetune", default="OTiS/otis_base.pth", help="finetune from checkpoint"
    )
    parser.add_argument("--global_pool", action="store_true", default=False)
    parser.add_argument("--attention_pool", action="store_true", default=False)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    parser.add_argument(
        "--ignore_pos_embed_y",
        action="store_true",
        default=False,
        help="Ignore pre-trained position embeddings Y (spatial axis) from checkpoint",
    )
    parser.add_argument(
        "--freeze_pos_embed_y",
        action="store_true",
        default=False,
        help="Make position embeddings Y (spatial axis) non-trainable",
    )

    # Dataset parameters
    parser.add_argument(
        "--downstream_task",
        default="classification",
        type=str,
        help="downstream task (default: classification)",
    )
    eval_criterions = [
        "epoch",
        "loss",
        "acc",
        "acc_balanced",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "cohen",
        "avg",
        "rmse",
        "mae",
        "pcc",
        "r2",
    ]
    parser.add_argument(
        "--eval_criterion",
        default="auroc",
        type=str,
        choices=eval_criterions,
        help="downstream task evaluation metric (default: auroc)",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="dataset path (default: None)",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        help="labels path (default: None)",
    )
    parser.add_argument(
        "--labels_mask_path", default="", type=str, help="labels path (default: None)"
    )

    parser.add_argument("--seconds", default=3, type=int, help="ECG Seconds")
    parser.add_argument("--fs", default=500, type=int, help="ECG samples per second")
    parser.add_argument(
        "--severity", default="70", type=str, help="severity of stenosis"
    )

    parser.add_argument(
        "--val_data_path",
        type=str,
        help="validation dataset path (default: None)",
    )
    parser.add_argument(
        "--val_labels_path",
        type=str,
        help="validation labels path (default: None)",
    )
    parser.add_argument(
        "--val_labels_mask_path",
        default="",
        type=str,
        help="validation labels path (default: None)",
    )

    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="test dataset path (default: None)",
    )
    parser.add_argument(
        "--test_labels_path",
        type=str,
        help="test labels path (default: None)",
    )
    parser.add_argument(
        "--test_labels_mask_path",
        default="",
        type=str,
        help="test labels path (default: None)",
    )

    parser.add_argument(
        "--lower_bnd", type=int, default=0, metavar="N", help="lower_bnd"
    )
    parser.add_argument(
        "--upper_bnd", type=int, default=0, metavar="N", help="upper_bnd"
    )

    parser.add_argument(
        "--nb_classes", default=2, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--pos_label",
        default=0,
        type=int,
        help="classification type with the smallest count",
    )

    parser.add_argument(
        "--output_dir", default=f"", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--log_dir", default="", help="path where to tensorboard log (default: ./logs)"
    )
    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument(
        "--head_layers", default=1, type=int, help="number of cls layers"
    )
    parser.add_argument(
        "--wandb_entity", default="", type=str, help="entity of the current run"
    )
    parser.add_argument(
        "--wandb_project",
        default="otis-ecg",
        type=str,
        help="project where to wandb log",
    )
    parser.add_argument(
        "--wandb_id", default="", type=str, help="id of the current run"
    )
    parser.add_argument("--suffix", default="", type=str, help="id suffix")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--save_logits", action="store_true", default=False, help="save model logits"
    )

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--extract_embeddings",
        action="store_true",
        default=False,
        help="Perform embeddings extraction only",
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor)",
    )

    return parser


def main(args):
    baseline_model = args.model.startswith("baseline")
    if args.model.endswith("ptacl"):
        args.seconds = 5
    if baseline_model:
        args.wandb_project = "ecg_baselines"
        if args.multilabel:
            args.wandb_project += "_echonext"
    args.severity = literal_eval(args.severity)
    args.unfreeze = literal_eval(args.unfreeze)
    args.time_steps = args.fs * args.seconds
    args.input_size = (args.input_channels, args.input_variates, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)

    timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
    if not args.eval:
        run_id = f'{timestamp}_{"".join(random.choices(string.ascii_lowercase, k=4))}_{str(args.lr)}_{str(args.severity).replace(",", "_")}{args.suffix}'
        if args.lin_probe:
            run_id += "_linprobe"
        if not args.finetune.startswith("OTiS/otis"):
            run_id += "_pretrained"

        if "unnormalized" in args.data_path:
            run_id += "_unnormalized"
        if baseline_model:
            run_id += args.model
        args.wandb_id = run_id
        args.output_dir = f"training_runs/{run_id}"
        os.makedirs(args.output_dir, exist_ok=True)
    print(f"cuda devices: {torch.cuda.device_count()}")
    misc.init_distributed_mode(args)

    # wandb logging
    if args.wandb == True and misc.is_main_process():
        config = vars(args)
        if args.wandb_id:
            wandb.init(
                project=args.wandb_project,
                id=args.wandb_id,
                config=config,
                entity=args.wandb_entity,
            )
        else:
            wandb.init(
                project=args.wandb_project, config=config, entity=args.wandb_entity
            )

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    print(f"rank: {misc.get_rank()}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    dataset_train = TimeSeriesDataset(
        data_path=args.data_path,
        labels_path=args.labels_path,
        labels_mask_path=args.labels_mask_path,
        downstream_task=args.downstream_task,
        weighted_loss=args.weighted_loss,
        univariate=args.univariate,
        num_samples=args.num_samples,
        dataset_percentage=args.dataset_percentage,
        train=True,
        augmentation=args.augmentation,
        N_val=1,
        multilabel=args.multilabel,
        severity=args.severity,
        seconds=args.seconds,
        fs=args.fs,
        args=args,
    )
    dataset_val = TimeSeriesDataset(
        data_path=args.val_data_path,
        labels_path=args.val_labels_path,
        labels_mask_path=args.val_labels_mask_path,
        downstream_task=args.downstream_task,
        domain_offsets=dataset_train.offsets,
        univariate=args.univariate,
        num_samples=args.num_samples,
        train=False,
        test=True,
        N_val=1,
        multilabel=args.multilabel,
        severity=args.severity,
        seconds=args.seconds,
        fs=args.fs,
        args=args,
    )
    dataset_test = TimeSeriesDataset(
        data_path=args.test_data_path,
        labels_path=args.test_labels_path,
        labels_mask_path=args.test_labels_mask_path,
        downstream_task=args.downstream_task,
        domain_offsets=dataset_train.offsets,
        univariate=args.univariate,
        num_samples=args.num_samples,
        train=False,
        test=True,
        N_val=1,
        multilabel=args.multilabel,
        severity=args.severity,
        seconds=args.seconds,
        fs=args.fs,
        args=args,
    )

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    print("Test set size: ", len(dataset_test))
    log_set_distribution(dataset_train, "Train")
    log_set_distribution(dataset_val, "Valid")
    log_set_distribution(dataset_test, "Test")

    # tensorboard logging
    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_train.collate_fn_ft,
        pin_memory=args.pin_mem,
        drop_last=len(dataset_train) % args.batch_size == 1,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_val.collate_fn_ft,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_test.collate_fn_ft,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    grad_updates = 0
    if args.num_grad_updates > 0:
        args.epochs = (
            (args.accum_iter * args.num_grad_updates) // len(data_loader_train)
        ) + 1
        args.validation_frequency = 100 / args.dataset_percentage
        wandb.config.update(
            {"epochs": args.epochs, "validation_frequency": args.validation_frequency},
            allow_val_change=True,
        )
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    args.nb_classes = data_loader_train.dataset.nb_classes
    print(f"Number of classes: {args.nb_classes}")
    print(f"mixup {mixup_active}")
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    if baseline_model:
        model = Baseline(
            args.model,
            lin_probe=args.lin_probe,
            nb_classes=args.nb_classes,
            head_layers=args.head_layers,
            pretrained_weights=args.resume,
        )
    else:
        model = models_vit.__dict__[args.model](
            domains=dataset_train.domains,
            img_size=args.input_size,
            patch_size=args.patch_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
            attention_pool=args.attention_pool,
            masking_blockwise=args.masking_blockwise,
            mask_ratio=args.mask_ratio,
            mask_c_ratio=args.mask_c_ratio,
            mask_t_ratio=args.mask_t_ratio,
        )

    if not baseline_model and args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu", weights_only=False)

        print("Load pretrained checkpoint from: %s" % args.finetune)
        if "clip" in args.finetune:
            if "ecg_model" in checkpoint:
                checkpoint = checkpoint["ecg_model"]
            checkpoint_model = {
                k.lstrip("model."): v
                for k, v in checkpoint.items()
                if k.startswith("model")
            }
        else:
            checkpoint_model = checkpoint["model"]

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # check if new and old patch_size match
        nb_channels_ckpt = checkpoint_model["patch_embed.proj.weight"].shape[-3]
        nb_channels_model = args.input_size[0]

        checkpoint_patch_size = checkpoint_model["patch_embed.proj.weight"].shape[-2:]
        patch_height_ckpt, patch_width_ckpt = (
            checkpoint_patch_size[0],
            checkpoint_patch_size[1],
        )
        patch_height_model, patch_width_model = args.patch_size[0], args.patch_size[1]

        if (
            nb_channels_ckpt != nb_channels_model
            or patch_height_ckpt != patch_height_model
            or patch_width_ckpt != patch_width_model
        ):
            # initialize new patch_embed
            for key in [
                "patch_embed.proj.weight",
                "patch_embed.proj.bias",
                "patch_embed.norm.weight",
                "patch_embed.norm.bias",
            ]:
                if key in checkpoint_model:
                    print(f"Removing key {key} from pretrained checkpoint")
                    del checkpoint_model[key]
            print("Initializing new patch_embed")

        # load pos_embed_x
        interpolate_pos_embed_x(model, checkpoint_model)

        key = "pos_embed_x"
        if key in checkpoint_model:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]

        # load pos_embed_y together with domain_offsets
        print(f"Identified domain: {dataset_train.domains}")
        assert (
            len(dataset_train.domains) == 1
        ), "There is more than one domain in the target dataset"
        target_domain = list(dataset_train.domains.keys())[0]
        # target_shape = list(dataset_train.domains.values())[0]

        pos_embed_y_available = False
        if "domains" not in checkpoint:
            checkpoint["domains"] = {"ecg": torch.Size(list(args.input_size))}
            checkpoint["domain_offsets"] = {"ecg": 4}

        checkpoint_domains = checkpoint["domains"]
        for domain, shape in checkpoint_domains.items():
            if domain == target_domain:  # and shape[1] == target_shape[1]:
                pos_embed_y_available = True
                break
        if (
            len(checkpoint["domain_offsets"]) > 1
            and sum([v for v in checkpoint["domain_offsets"].values()]) == 0
        ):
            # domain-agnostic pos_embed_y
            print("INFO: Found domain-agnostic pos_embed_y in checkpoint")
            pos_embed_y_available = True  # if pos_embed_y_available = False before

            # set offset to zero
            print(dataset_train.domain)
            checkpoint["domain_offsets"][dataset_train.domain[0][0]] = 0

        if not args.ignore_pos_embed_y and pos_embed_y_available:
            print("Loading pos_embed_y from checkpoint")
            print(f"Current pos_embed_y shape: {model.pos_embed_y.weight.shape}")
            model.pos_embed_y = None
            model.pos_embed_y = torch.nn.Embedding.from_pretrained(
                checkpoint_model["pos_embed_y.weight"]
            )
            print(f"New pos_embed_y shape: {model.pos_embed_y.weight.shape}")

            # load domain_offsets
            dataset_train.set_domain_offsets(checkpoint["domain_offsets"])
            dataset_val.set_domain_offsets(checkpoint["domain_offsets"])
            dataset_test.set_domain_offsets(checkpoint["domain_offsets"])
        else:
            print("Initializing new pos_embed_y")

        key = "pos_embed_y.weight"
        if key in checkpoint_model:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]
        # load pretrained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(msg)

        if args.global_pool:
            assert {
                "head.weight",
                "head.bias",
                "fc_norm.weight",
                "fc_norm.bias",
                "pos_embed_x",
                "pos_embed_y.weight",
            }.issubset(set(msg.missing_keys))
        elif args.attention_pool:
            assert {
                "head.weight",
                "head.bias",
                "fc_norm.weight",
                "fc_norm.bias",
                "pos_embed_x",
                "pos_embed_y.weight",
            }.issubset(set(msg.missing_keys))
        else:
            assert {
                "head.weight",
                "head.bias",
                "pos_embed_x",
                "pos_embed_y.weight",
            }.issubset(set(msg.missing_keys))

    if not baseline_model:
        if args.eval:
            in_features = 192
        else:
            in_features = model.head.in_features

        layers = [torch.nn.BatchNorm1d(in_features, affine=False, eps=1e-6)]
        for _ in range(args.head_layers - 1):
            layers.extend([nn.Linear(in_features, in_features), nn.ReLU()])
        layers.append(nn.Linear(in_features, args.nb_classes))
        model.head = torch.nn.Sequential(*layers)

        for layer in model.head:
            if isinstance(layer, nn.Linear):
                trunc_normal_(layer.weight, std=0.01)

        if args.freeze_pos_embed_y:
            print(f"Freeze pos_embed_y")
            model.pos_embed_y.weight.requires_grad = False
        else:
            print(f"Unfreeze pos_embed_y")
            model.pos_embed_y.weight.requires_grad = True
    if args.lin_probe or args.freeze_backbone:
        # freeze all but the head
        for _, p in model.named_parameters():
            p.requires_grad = False
        for n, p in model.head.named_parameters():
            print(f"Unfreeze head.{n}")
            p.requires_grad = True
        if args.freeze_backbone:
            model.cls_token.requires_grad = True

    if baseline_model and args.eval:
        sub_strings = args.resume.split("/")
        if "checkpoint" in sub_strings[-1] or "last.pth" == sub_strings[-1]:
            nb_ckpts = 1
        else:
            nb_ckpts = int(sub_strings[-1]) + 1

        if "checkpoint" not in sub_strings[-1] and "last.pth" != sub_strings[-1]:
            args.resume = "/".join(sub_strings[:-1]) + "/checkpoint-" + str(0) + ".pth"

        # load pos_embed_x from checkpoint
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint["model"]
        interpolate_pos_embed_x(model, checkpoint_model)

        # load pos_embed_y from checkpoint
        print(f"Current pos_embed_y shape: {model.pos_embed_y.weight.shape}")
        model.pos_embed_y = None
        model.pos_embed_y = torch.nn.Embedding.from_pretrained(
            checkpoint_model["pos_embed_y.weight"]
        )
        print(f"New pos_embed_y shape: {model.pos_embed_y.weight.shape}")

        # load domain_offsets
        print(f"Current domain_offsets: {dataset_train.offsets}")
        dataset_train.set_domain_offsets(checkpoint["domain_offsets"])
        dataset_val.set_domain_offsets(checkpoint["domain_offsets"])
        dataset_test.set_domain_offsets(checkpoint["domain_offsets"])

        print(f"New domain_offsets: {dataset_train.offsets}")

    model.to(device, non_blocking=True)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 32

    print("base lr: %.2e" % (args.lr * 32 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    optimizer = create_optimizer_with_different_lrs(
        model_without_ddp,
        head_lr=args.lr,
        backbone_lr=args.backbone_lr,
        wd=args.weight_decay,
    )
    loss_scaler = NativeScaler()

    class_weights = None
    if dataset_train.class_weights is not None:
        class_weights = dataset_train.class_weights.to(device=device, non_blocking=True)

    if args.multilabel:
        criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
    elif args.smoothing > 0.0:
        criterion = torch.nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=args.smoothing
        )
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("criterion = %s" % str(criterion))

    if not baseline_model and not args.eval:
        misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )

    if args.eval:
        sub_strings = args.resume.split("/")
        if "checkpoint" in sub_strings[-1] or "last.pth" == sub_strings[-1]:
            nb_ckpts = 1
        else:
            nb_ckpts = int(sub_strings[-1]) + 1

        for epoch in range(0, nb_ckpts):
            if "checkpoint" not in sub_strings[-1] and "last.pth" != sub_strings[-1]:
                args.resume = (
                    "/".join(sub_strings[:-1]) + "/checkpoint-" + str(epoch) + ".pth"
                )
            if not baseline_model:
                misc.load_model(
                    args=args,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                )

            test_stats, test_history = evaluate(
                data_loader_val,
                model_without_ddp,
                device,
                epoch,
                log_writer=log_writer,
                args=args,
            )
            if args.downstream_task == "classification":
                print(
                    f"Accuracy / Accuracy (balanced) / Precision / Recall / F1 / AUROC / Cohen's Kappa of the network on {len(dataset_val)} test samples: ",
                    f"{test_stats['acc']:.2f}% / {test_stats['acc_balanced']:.2f}% / {test_stats['precision']:.2f}% / {test_stats['recall']:.2f}% / ",
                    f"{test_stats['f1']:.2f}% / {test_stats['auroc']:.2f}% / {test_stats['cohen']:.4f}",
                )
            elif args.downstream_task == "regression":
                print(
                    f"Root Mean Squared Error (RMSE) / Mean Absolute Error (MAE) / Pearson Correlation Coefficient (PCC) / R Squared (R2) ",
                    f"of the network on {len(dataset_val)} test samples: {test_stats['rmse']:.4f} / {test_stats['mae']:.4f} / ",
                    f"{test_stats['pcc']:.4f} / {test_stats['r2']:.4f}",
                )

            if args.wandb and misc.is_main_process():
                wandb.log({f"test/{k}": v for k, v in test_history.items()})

        exit(0)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    print(f"Start training for {args.epochs} epochs")

    best_stats = {
        "loss": np.inf,
        "acc": 0.0,
        "acc_balanced": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auroc": 0.0,
        "auprc": 0.0,
        "cohen": 0.0,
        "avg": 0.0,
        "epoch": 0,
        "rmse": np.inf,
        "mae": np.inf,
        "pcc": 0.0,
        "r2": -1.0,
    }
    best_eval_scores = {
        "count": 1,
        "nb_ckpts_max": 3,
        "eval_criterion": [best_stats[args.eval_criterion]],
    }
    epochs_to_unfreeze_at = [k[0] for k in args.unfreeze]
    blocks_to_unfreeze = [k[1] for k in args.unfreeze]

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if epoch in epochs_to_unfreeze_at:
            model.unfreeze_last_n_blocks(
                blocks_to_unfreeze[epochs_to_unfreeze_at.index(epoch)]
            )
            head_lr = [g["lr"] for g in optimizer.param_groups if g["part"] == "head"][
                0
            ]
            optimizer = create_optimizer_with_different_lrs(
                model,
                head_lr=head_lr,
                backbone_lr=args.backbone_lr,
                wd=args.weight_decay,
            )
        train_stats, train_history, grad_updates = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            grad_updates=grad_updates,
            args=args,
        )
        print(f"{grad_updates}/{args.num_grad_updates} Gradient Updates Done!")
        if not (epoch % args.validation_frequency == 0 or epoch == args.epochs - 1):
            test_history = {}
        else:
            test_stats, test_history = evaluate(
                data_loader_val,
                model_without_ddp,
                device,
                epoch,
                log_writer=log_writer,
                args=args,
            )

            test_stats["avg"] = (
                test_stats["acc_balanced"] + test_stats["f1"] + test_stats["cohen"]
            ) / 3

            if (
                early_stop.evaluate_increasing_metric(
                    val_metric=test_stats[args.eval_criterion]
                )
                and misc.is_main_process()
            ):
                print("Early stopping the training")
                break

            if args.output_dir and test_stats[args.eval_criterion] >= min(
                best_eval_scores["eval_criterion"]
            ):
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores["count"] < best_eval_scores["nb_ckpts_max"]:
                    best_eval_scores["count"] += 1
                else:
                    best_eval_scores["eval_criterion"] = sorted(
                        best_eval_scores["eval_criterion"], reverse=True
                    )
                    best_eval_scores["eval_criterion"].pop()
                best_eval_scores["eval_criterion"].append(
                    test_stats[args.eval_criterion]
                )

                misc.save_best_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    test_stats=test_stats,
                    evaluation_criterion=args.eval_criterion,
                    nb_ckpts_max=best_eval_scores["nb_ckpts_max"],
                    mode="increasing",
                    domains=dataset_train.domains,
                    domain_offsets=dataset_train.offsets,
                )

            test_history["test_avg"] = test_stats["avg"]

            best_stats["loss"] = min(best_stats["loss"], test_stats["loss"])

            if args.downstream_task == "classification":
                # update best stats

                best_stats["f1"] = max(best_stats["f1"], test_stats["f1"])
                best_stats["precision"] = max(
                    best_stats["precision"], test_stats["precision"]
                )
                best_stats["recall"] = max(best_stats["recall"], test_stats["recall"])
                best_stats["acc"] = max(best_stats["acc"], test_stats["acc"])
                best_stats["acc_balanced"] = max(
                    best_stats["acc_balanced"], test_stats["acc_balanced"]
                )
                best_stats["cohen"] = max(best_stats["cohen"], test_stats["cohen"])

                if test_stats["avg"] >= best_stats["avg"]:
                    best_stats["epoch"] = epoch
                best_stats["avg"] = max(best_stats["avg"], test_stats["avg"])

                print(
                    f"Accuracy / Accuracy (balanced) / Precision / Recall / F1 / AUROC/ Cohen's Kappa of the network on {len(dataset_val)} test samples: ",
                    f"{test_stats['acc']:.2f}% / {test_stats['acc_balanced']:.2f}% / {test_stats['precision']:.2f}% / {test_stats['recall']:.2f}% / ",
                    f"{test_stats['f1']:.2f}% / {test_stats['auroc']:.2f}% / {test_stats['cohen']:.4f}",
                )
                print(
                    f'Max Accuracy / Accuracy (balanced) / Precision / Recall / F1 : {best_stats["acc"]:.2f}% / ',
                    f'{best_stats["acc_balanced"]:.2f}% / {best_stats["precision"]:.2f}% / {best_stats["recall"]:.2f}% / {best_stats["f1"]:.2f}% / ',
                    f'{best_stats["cohen"]:.4f}\n',
                )

        total_time = time.time() - start_time
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
            "epoch": epoch,
            "time_per_epoch": total_time,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.wandb and misc.is_main_process():
            format_key = lambda s, k: f"{s}_{k}" if "/" in k else f"{s}/{k}"
            wandb.log({format_key("train", k): v for k, v in train_history.items()})
            wandb.log({format_key("val", k): v for k, v in test_history.items()})
            wandb.log({"Time per epoch [sec]": total_time})
        if grad_updates == args.num_grad_updates:
            print(f"{grad_updates} grad updates done!")
            break

    misc.save_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        epoch=epoch,
        nb_ckpts_max=best_eval_scores["nb_ckpts_max"],
        domains=dataset_train.domains,
        domain_offsets=dataset_train.offsets,
        last=True,
    )

    if args.test and misc.is_main_process():
        for ind, checkpoint_to_load in enumerate(["last", "best"]):
            if checkpoint_to_load == "last":
                args.resume = os.path.join(args.output_dir, "last.pth")
            else:
                args.resume = misc.get_best_ckpt(
                    args.output_dir, eval_criterion=args.eval_criterion
                )

            checkpoint = torch.load(args.resume, weights_only=False)
            model_without_ddp.load_state_dict(checkpoint["model"])
            print("Run test data on checkpoint model %s" % args.resume)

            test_stats, test_history = evaluate(
                data_loader_test,
                model_without_ddp,
                device,
                epoch=-1,
                log_writer=log_writer,
                args=args,
            )
            actual_test_history = {}
            for k, v in test_history.items():
                key = k
                if k.endswith(
                    ("f1", "precision", "recall", "acc", "acc_balanced", "auroc")
                ):
                    actual_test_history[key] = v
            if args.wandb and misc.is_main_process():
                format_key = lambda s, k: f"{s}_{k}" if "/" in k else f"{s}/{k}"
                wandb.log(
                    {
                        format_key("actual_test_history", k): v
                        for k, v in actual_test_history.items()
                    }
                )
                wandb.log({"actual_test_history/eval_ind": ind})

    if args.wandb and misc.is_main_process():
        wandb.log({f"Best Statistics/{k}": v for k, v in best_stats.items()})
        wandb.finish()

    exit(0)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
