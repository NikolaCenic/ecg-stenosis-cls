from stenosis_classification.clip.dataset import EcgAngioDataset
from stenosis_classification.clip.loss import CLIPLoss
from stenosis_classification.clip.utils import (
    create_optimizer,
    create_loaders,
    log_clip_logits,
    compute_clip_metrics,
)
from stenosis_classification.otis.engine_finetune import log_12lead_ecg_to_wandb
from stenosis_classification.clip.models import (
    ECGModelWrapper,
    AngioModelWrapper,
    CLIPSegmentMIL,
    count_params,
)
from stenosis_classification.static_classification.utils import (
    set_seed,
    compute_metrics,
)
from stenosis_classification.static_classification.loss import WeightedBCE
from stenosis_classification.clip.knn_online_eval import knn_online_eval
from torch.utils.data import DataLoader
import hydra
import logging
import torch
import wandb
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from omegaconf import DictConfig, OmegaConf
from ignite.handlers.checkpoint import ModelCheckpoint

import warnings
import datetime
import random
import string
import os
import shutil

# Get the current timestamp


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def pad_probs(probs, batch_size):
    if len(probs) < batch_size:
        paded_probs = torch.zeros(batch_size, batch_size).to(probs.device)
        paded_probs[: len(probs), : len(probs)] = probs
        return paded_probs
    return probs


@hydra.main(config_path=".", config_name="config.yaml", version_base="1.3")
def pretrain(cfg: DictConfig):
    cfg.clip_mil_training.do_stenosis_training = (
        cfg.clip_mil_training.do_stenosis_training
        or cfg.clip_mil_training.stenosis_loss_lambda > 0
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = f'{timestamp}_{"".join(random.choices(string.ascii_letters, k=4))}_{cfg.ecg.model.type}{f"_MIL-CLIP_{cfg.clip_mil_training.stenosis_loss_lambda}" if cfg.clip_mil_training.do_stenosis_training else ""}{cfg.get("run_suffix", "")}'

    model_dir = os.path.join("clip-models", run_name)

    knn_eval_dir = os.path.join(model_dir, "knn")
    os.makedirs(knn_eval_dir)
    knn_eval_checkpointer = ModelCheckpoint(
        knn_eval_dir,
        filename_prefix="knn_best",
        score_function=lambda x: x["AUC"],
        score_name="auc",
        n_saved=3,
    )

    set_seed(cfg.seed)
    # setup out
    batch_sizes = {
        "otis": {
            3: 128,
            5: None,
            10: None,
        },
        "echoing-ecg": {
            3: 128,
            5: None,
            10: None,
        },
        "moment": {
            3: None,
            5: None,
            10: None,
        },
    }
    cfg.batch_size = batch_sizes[cfg.ecg.model.type][cfg.ecg.seconds]
    cfg.ecg.model.input_size = [1, 12, cfg.ecg.seconds * 500]
    cfg.angio.multiframe.frames_per_view = (
        1 + cfg.angio.multiframe.right + cfg.angio.multiframe.left
    )

    device = torch.device(cfg.device)

    ecg_model = ECGModelWrapper(cfg.ecg.model, cfg.embed_dim).to(device)
    cfg.ecg.model.ecg_pos_emb_offset = ecg_model.ecg_pos_emb_offset

    ecg_optim, ecg_lr_scheduler = create_optimizer(
        model=ecg_model, epochs=cfg.epochs, cfg=cfg.ecg.optim
    )
    if cfg.clip_mil_training.do_stenosis_training:
        print("--------------------------------------------------------------------")
        print("Setting up stenosis training in CLIP pipeline.")
        angio_model = CLIPSegmentMIL(
            clip_embed_dim=cfg.embed_dim,
            frames_per_view=1,
        ).to(device)
        stenosis_criterion = WeightedBCE(
            device=device,
            frequency_weights=torch.ones(11),
            loss_config=cfg.clip_mil_training.loss,
        )
        print("--------------------------------------------------------------------")

    else:
        angio_model = AngioModelWrapper(cfg.angio.model, cfg.embed_dim).to(device)
    angio_optim, angio_lr_scheduler = create_optimizer(
        model=angio_model, epochs=cfg.epochs, cfg=cfg.angio.optim
    )
    if cfg.compile:
        ecg_model.compile()
        angio_model.compile()

    train_loader, val_loader = create_loaders(cfg)

    logger.info(
        f"Train Loader size: {len(train_loader)} (total samples {len(train_loader.dataset)})"
    )
    logger.info(
        f"Valid Loader size: {len(val_loader)} (total samples {len(val_loader.dataset)})"
    )
    wandb.init(
        project="CLIP",
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    criterion = CLIPLoss(temperature=cfg.loss.temperature)
    scaler = GradScaler()
    try:
        for epoch in range(cfg.epochs):
            ecg_model.train()
            angio_model.train()
            train_logits = []
            stenosis_train_logits = []
            stenosis_train_targets = []
            for batch_ind, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Train Epoch {epoch+1}",
            ):
                (
                    ecg,
                    ecg_pos_emb,
                    angio,
                    samples_cnt,
                    angulations,
                    angio_keys,
                    stenosis_targets,
                ) = data

                ecg = ecg.to(device, non_blocking=True)
                ecg_pos_emb = ecg_pos_emb.to(device, non_blocking=True)
                angio = angio.to(device, non_blocking=True)

                with autocast(dtype=torch.float16):
                    ecg_emb = ecg_model(ecg, ecg_pos_emb)
                    if cfg.clip_mil_training.do_stenosis_training:
                        stenosis_targets = stenosis_targets.to(device)
                        angio_emb, stenosis_logits = angio_model(angio, samples_cnt)
                        stenosis_loss, stenosis_loss_dict = stenosis_criterion(
                            stenosis_logits, stenosis_targets
                        )

                        wandb.log(
                            {
                                **{
                                    f"Stenosis Train/{k}": v
                                    for k, v in stenosis_loss_dict.items()
                                },
                                "Stenosis Train/loss": stenosis_loss.item(),
                                f"Stenosis Train/iteration": epoch * len(train_loader)
                                + batch_ind,
                            }
                        )

                        stenosis_train_logits.extend(
                            list(stenosis_logits.cpu().detach())
                        )
                        stenosis_train_targets.extend(
                            list(stenosis_targets.cpu().detach())
                        )
                    else:
                        angio_emb = angio_model(angio, samples_cnt, angulations)
                        stenosis_loss = 0

                    clip_loss, clip_logits = criterion(
                        ecg_emb=ecg_emb, angio_emb=angio_emb
                    )
                total_loss = clip_loss
                if cfg.clip_mil_training.do_stenosis_training:
                    total_loss += (
                        cfg.clip_mil_training.stenosis_loss_lambda * stenosis_loss
                    )

                ecg_optim.zero_grad(set_to_none=True)
                angio_optim.zero_grad(set_to_none=True)

                scaler.scale(total_loss).backward()
                scaler.unscale_(ecg_optim)
                scaler.unscale_(angio_optim)
                scaler.step(ecg_optim)
                scaler.step(angio_optim)
                scaler.update()

                if cfg.batch_size == len(clip_logits):
                    wandb.log(
                        {
                            "CLIP Train/clip loss": clip_loss.item(),
                            "CLIP Train/total loss": total_loss.item(),
                        }
                    )

                clip_logits = pad_probs(clip_logits, cfg.batch_size)
                train_logits.extend([clip_logits, clip_logits.T])

                if batch_ind == 0 and epoch % cfg.log_period == 0:
                    print(
                        f"SHAPES Angio: {angio.shape}->{angio_emb.shape}, ECG: {ecg.shape}->{ecg_emb.shape}"
                    )

                    log_clip_logits(clip_logits, "CLIP Train")
                    wandb.log(
                        {
                            "CLIP Train/12lead": [
                                log_12lead_ecg_to_wandb(e[0]) for e in ecg[:2]
                            ]
                        }
                    )

            ecg_model.eval()
            angio_model.eval()
            val_logits = []
            stenosis_val_logits = []
            stenosis_val_targets = []
            for batch_ind, data in tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Valid Epoch {epoch+1}",
            ):
                with torch.no_grad():
                    (
                        ecg,
                        ecg_pos_emb,
                        angio,
                        samples_cnt,
                        angulations,
                        angio_keys,
                        stenosis_targets,
                    ) = data

                    ecg = ecg.to(device, non_blocking=True)
                    ecg_pos_emb = ecg_pos_emb.to(device, non_blocking=True)
                    angio = angio.to(device, non_blocking=True)
                    with autocast(dtype=torch.float16):
                        ecg_emb = ecg_model(ecg, ecg_pos_emb)
                        if cfg.clip_mil_training.do_stenosis_training:
                            stenosis_targets = stenosis_targets.to(device)

                            angio_emb, stenosis_logits = angio_model(angio, samples_cnt)
                            stenosis_loss, stenosis_loss_dict = stenosis_criterion(
                                stenosis_logits, stenosis_targets
                            )

                            wandb.log(
                                {
                                    **{
                                        f"Stenosis Valid/{k}": v
                                        for k, v in stenosis_loss_dict.items()
                                    },
                                    "Stenosis Valid/loss": stenosis_loss,
                                    f"Stenosis Valid/iteration": epoch
                                    * len(train_loader)
                                    + batch_ind,
                                }
                            )

                            stenosis_val_logits.extend(
                                list(stenosis_logits.cpu().detach())
                            )
                            stenosis_val_targets.extend(
                                list(stenosis_targets.cpu().detach())
                            )
                        else:
                            angio_emb = angio_model(angio, samples_cnt, angulations)

                        clip_loss, clip_logits = criterion(
                            ecg_emb=ecg_emb, angio_emb=angio_emb
                        )

                    if cfg.batch_size == len(clip_logits):
                        wandb.log({"CLIP Valid/loss": clip_loss.item()})
                        stenosis_loss = 0

                    clip_logits = pad_probs(clip_logits, cfg.batch_size)

                    val_logits.extend([clip_logits, clip_logits.T])

                    if batch_ind == 0 and epoch % cfg.log_period == 0:
                        log_clip_logits(clip_logits, "CLIP Valid")
                        wandb.log(
                            {
                                "CLIP Valid/12lead": [
                                    log_12lead_ecg_to_wandb(e[0]) for e in ecg[:2]
                                ]
                            }
                        )

            ecg_lr_scheduler.step()
            angio_lr_scheduler.step()

            dict_to_log = {
                "angio_lr": angio_optim.param_groups[0]["lr"],
                "angio_trainable_params": count_params(angio_model),
                "ecg_lr": ecg_optim.param_groups[0]["lr"],
                "ecg_trainable_params": count_params(ecg_model),
                "epoch": epoch,
                **compute_clip_metrics(train_logits, "CLIP Train"),
                **compute_clip_metrics(val_logits, "CLIP Valid"),
            }

            if cfg.clip_mil_training.do_stenosis_training:

                train_metrics, _ = compute_metrics(
                    logits=stenosis_train_logits,
                    targets=stenosis_train_targets,
                    threshold=0.5,
                    compute_segment_metrics=False,
                )
                val_metrics, _ = compute_metrics(
                    logits=stenosis_val_logits,
                    targets=stenosis_val_targets,
                    threshold=0.5,
                    compute_segment_metrics=False,
                )

                train_prefix = lambda k: (
                    f"Stenosis Train/{k}" if "/" not in k else f"Train_{k}"
                )
                dict_to_log.update(
                    {train_prefix(k): v for k, v in train_metrics.items()}
                )

                valid_prefix = lambda k: (
                    f"Stenosis Valid/{k}" if "/" not in k else f"Valid__{k}"
                )
                dict_to_log.update({valid_prefix(k): v for k, v in val_metrics.items()})

            wandb.log(dict_to_log)

            if cfg.online_eval.period and epoch % cfg.online_eval.period == 0:
                knn_metrics = {
                    "step": epoch / cfg.online_eval.period,
                    **knn_online_eval(model=ecg_model, cfg=cfg),
                }

                knn_eval_checkpointer(
                    knn_metrics, {"ecg_model": ecg_model, "angio_model": angio_model}
                )
                wandb.log({f"KNN/Val {k}": v for k, v in knn_metrics.items()})

    finally:
        if len(os.listdir(model_dir)) == 0:
            shutil.rmtree(model_dir)
            wandb.finish()
        else:
            OmegaConf.save(cfg, os.path.join(model_dir, "config.yaml"))
            torch.save(
                {
                    "ecg_model": ecg_model.state_dict(),
                    "angio_model": angio_model.state_dict(),
                },
                os.path.join(model_dir, "ecg_last.pt"),
            )
            wandb.finish()


if __name__ == "__main__":
    pretrain()
