from stenosis_classification.clip.dataset import EcgAngioDataset
from stenosis_classification.static_classification.models import SegmentMIL
from stenosis_classification.static_classification.utils import get_segments_to_use
from stenosis_classification.otis import models_vit
from stenosis_classification.otis.util.pos_embed import interpolate_pos_embed_x
import torch

from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import os
from omegaconf import DictConfig
from timm.models.layers import trunc_normal_
import numpy as np
import matplotlib.pyplot as plt
import wandb

from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def create_loaders(cfg: DictConfig):
    train_loader = DataLoader(
        EcgAngioDataset(
            setting="train",
            precomputed_angio_embeddings=cfg.angio.model.precomputed_embeddings,
            use_patch_embeddings=cfg.clip_mil_training.do_stenosis_training,
            angio_augmentation=cfg.angio.augmentation,
            ecg_augmentation=cfg.ecg.augmentation,
            num_samples=cfg.train_num_samples,
            ecg_patch_size=cfg.ecg.model.patch_size,
            ecg_pos_emb_offset=cfg.ecg.model.ecg_pos_emb_offset,
            multiframe=cfg.angio.multiframe,
            ecg_length_seconds=cfg.ecg.seconds,
            drop_views=cfg.angio.drop_views,
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=EcgAngioDataset.collate_fn_,
        num_workers=cfg.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        EcgAngioDataset(
            setting="val",
            precomputed_angio_embeddings=cfg.angio.model.precomputed_embeddings,
            use_patch_embeddings=cfg.clip_mil_training.do_stenosis_training,
            angio_augmentation=False,
            ecg_augmentation=None,
            num_samples=cfg.val_num_samples,
            ecg_patch_size=cfg.ecg.model.patch_size,
            ecg_pos_emb_offset=cfg.ecg.model.ecg_pos_emb_offset,
            multiframe=cfg.angio.multiframe,
            ecg_length_seconds=cfg.ecg.seconds,
            drop_views=False,
        ),
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=EcgAngioDataset.collate_fn_,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_optimizer(model, epochs, cfg):
    if count_params(model) == 0:
        return None, None

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=cfg.warmup_epochs
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - cfg.warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )
    return optimizer, scheduler


def has_weights(module: torch.nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def load_ecg_model(cfg: DictConfig):
    input_size = cfg.get("input_size", (1, 12, 500 * cfg.get("ecg_seconds", 10)))
    patch_size = cfg.get("patch_size", [1, 24])
    size = cfg.get("size", "base")

    model = models_vit.__dict__[f"vit_{size}Deep_patchX"](
        domains={"ecgs": input_size},
        img_size=input_size,
        patch_size=patch_size,
        num_classes=cfg.get("nb_classes", 0),
        drop_path_rate=cfg.get("drop_path", 0.1),
        global_pool=False,
        attention_pool=False,
        return_all_tokens=False,
        masking_blockwise=False,
        mask_ratio=0,
        mask_c_ratio=0,
        mask_t_ratio=0,
    )
    ecg_domain_offset = 0

    pretrained_path = f"../finetune/OTiS/otis_{size}.pth"
    checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if isinstance(cfg.pretrained, str) and "clip" in cfg.pretrained:
        checkpoint_model = {
            k.lstrip("model."): v
            for k, v in torch.load(cfg.pretrained, map_location="cpu")[
                "ecg_model"
            ].items()
            if k.startswith("model.")
        }
        print(f"CLIP Otis Loaded from {cfg.pretrained}")

    elif os.path.isfile(cfg.pretrained):
        checkpoint_model = torch.load(cfg.pretrained, map_location="cpu")
        print(f"Full Otis Loaded from {cfg.pretrained}")
    else:
        print(f"Otis Loaded from {pretrained_path}")
        checkpoint_model = checkpoint["model"]

    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # check if new and old patch_size match
    nb_channels_ckpt = checkpoint_model["patch_embed.proj.weight"].shape[-3]
    nb_channels_model = input_size[0]
    patch_height_ckpt, patch_width_ckpt = checkpoint_model[
        "patch_embed.proj.weight"
    ].shape[-2:]
    patch_height_model, patch_width_model = patch_size

    assert (
        nb_channels_ckpt == nb_channels_model
        and patch_height_ckpt == patch_height_model
        and patch_width_ckpt == patch_width_model
    )
    # load pos_embed_x
    interpolate_pos_embed_x(model, checkpoint_model)

    key = "pos_embed_x"
    if key in checkpoint_model:
        print(f"Removing key {key} from pretrained checkpoint")
        del checkpoint_model[key]

    # load pos_embed_y together with domain_offsets
    target_domain = "ecg"

    pos_embed_y_available = False
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
        checkpoint["domain_offsets"]["ecg"] = 0
    if not cfg.get("ignore_pos_embed_y", False) and pos_embed_y_available:
        print("Loading pos_embed_y from checkpoint")
        print(f"Current pos_embed_y shape: {model.pos_embed_y.weight.shape}")
        model.pos_embed_y = None
        model.pos_embed_y = torch.nn.Embedding.from_pretrained(
            checkpoint_model["pos_embed_y.weight"]
        )
        print(f"New pos_embed_y shape: {model.pos_embed_y.weight.shape}")
        ecg_domain_offset = checkpoint["domain_offsets"]["ecg"]
    else:
        print("Initializing new pos_embed_y")

    key = "pos_embed_y.weight"
    if key in checkpoint_model:
        print(f"Removing key {key} from pretrained checkpoint")
        del checkpoint_model[key]

    # load pretrained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    # manually initialize fc layer
    if has_weights(model.head) and not checkpoint_model.get("head.weight", False):
        trunc_normal_(model.head.weight, std=0.01)  # 2e-5)

    print(f"ECG Model - {count_params(model)}M params.")
    print("-----------------------------------------------")

    return model, ecg_domain_offset


def load_angio_model(cfg: DictConfig):
    if cfg.get("precomputed_embeddings", False):
        model = torch.nn.Identity()
    else:
        # not used
        pretrained_weights = {
            1: "SEGMENT_MIL_WEIGHTS_DIR",
            2: "SEGMENT_MIL_WEIGHTS_DIR",
            3: "SEGMENT_MIL_WEIGHTS_DIR",
            4: "SEGMENT_MIL_WEIGHTS_DIR",
            5: "SEGMENT_MIL_WEIGHTS_DIR",
        }
        model = SegmentMIL(
            backbone_type=cfg.get("backbone_type", "vit"),
            pretrained=True,
            encode_level=cfg.get("encode_level", "patch"),
            embed_dim=cfg.get("embed_dim", 512),
            resolution=cfg.get("resolution", 518),
            projector_layers=cfg.get("projector_layers", 0),
            classifier_layers=cfg.get("classifier_layers", 0),
            attention_type=cfg.get("attention_type", "transformer"),
            attention_heads=cfg.get("attention_heads", 4),
            shared_classifier=cfg.get("shared_classifier", False),
            hierarchical=cfg.get("hierarchical", True),
            transformer_layers=cfg.get("transformer_layers", 2),
            segments_to_use=get_segments_to_use(cfg.segments),
            frames_per_view=cfg.get("frames_per_view", 1),
            cnt_pos_embeddings=cfg.get("cnt_pos_embeddings", 256),
        )

        if cfg.get("pretrained", False):
            if isinstance(cfg.pretrained, bool):
                weights_path = os.path.join(
                    pretrained_weights[cfg.frames_per_view], "last.pt"
                )
            else:
                weights_path = cfg.pretrained
            msg = model.load_state_dict(torch.load(weights_path))
            print(msg)
            print(f"Model loaded from {weights_path}")
        print(f"ANGIO Model - {count_params(model)}M params.")
        print("-----------------------------------------------")

    return model


def log_clip_logits(clip_logits, prefix):

    clip_logits = clip_logits.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(clip_logits, cmap="viridis")
    N = len(clip_logits)

    cbar = fig.colorbar(im)
    cbar.set_label("Relative intensity within row")

    ax.set_xlabel("Classes")
    ax.set_ylabel("Samples")

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    plt.tight_layout()
    wandb.log(
        {
            f"{prefix}/logits_heatmap": wandb.Image(plt),
        }
    )


def compute_clip_metrics(clip_logits_per_batch, prefix):

    clip_logits = torch.cat(clip_logits_per_batch).detach().cpu()
    N, C = clip_logits.shape
    targets = torch.arange(C).repeat(N)[: len(clip_logits)]

    preds = torch.argmax(clip_logits, dim=1).numpy()

    y_true = targets.numpy()
    acc = accuracy_score(y_true, preds)

    f1 = f1_score(y_true, preds, average="macro")
    recall = recall_score(y_true, preds, average="macro")
    precision = precision_score(y_true, preds, average="macro")
    k = 5
    acc_topk = top_k_accuracy_score(y_true, clip_logits, k=k)
    try:
        auc = roc_auc_score(
            y_true,
            torch.nn.functional.softmax(clip_logits),
            multi_class="ovr",
            average="macro",
        )
    except Exception as e:
        print(e)
        auc = 0.0
    metrics = {
        "Accuracy": acc,
        f"Accuracy Top {k}": acc_topk,
        "F1_score": f1,
        "Recall": recall,
        "Precision": precision,
        "AUC": auc,
    }
    return {f"{prefix}/{k}": round(v, 3) for k, v in metrics.items()}


if __name__ == "__main__":
    p = torch.rand(10, 10)

    print(compute_clip_metrics(clip_logits_per_batch=[p, p], prefix=""))
