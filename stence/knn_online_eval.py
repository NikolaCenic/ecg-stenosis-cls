from stenosis_classification.otis.util.dataset import TimeSeriesDataset
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import hydra
from models import ECGModelWrapper


def create_loaders(cfg):
    args = OmegaConf.create(
        {
            "patch_size": cfg.ecg.model.patch_size,
            "time_steps": cfg.ecg.model.input_size[-1],
        }
    )
    train_dataset = TimeSeriesDataset(
        data_path="../finetune/ecg_data/stenosis/data/train_data.pt",
        labels_path="../finetune/ecg_data/stenosis/data/train_labels.pt",
        labels_mask_path=None,
        downstream_task="classification",
        domain_offsets={"ecg": 5},
        univariate=False,
        num_samples=-1,
        train=True,
        N_val=1,
        args=args,
        augmentation=False,
    )

    val_dataset = TimeSeriesDataset(
        data_path="../finetune/ecg_data/stenosis/data/val_data.pt",
        labels_path="../finetune/ecg_data/stenosis/data/val_labels.pt",
        labels_mask_path=None,
        downstream_task="classification",
        domain_offsets={"ecg": 5},
        univariate=False,
        num_samples=-1,
        train=False,
        N_val=1,
        args=args,
    )
    batch_size = 20
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=train_dataset.collate_fn_ft,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=val_dataset.collate_fn_ft,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, val_loader


def collect_embeddings_and_targets(loader, model, device, set):
    model.eval()
    embeddgins, targets = [], []

    for batch_ind, (ecg, labels, _, ecg_pos_emb) in tqdm(
        enumerate(loader), f"Collecting embeddings for {set} set:", total=len(loader)
    ):
        with torch.no_grad():
            ecg = ecg.to(device, non_blocking=True)
            ecg_pos_emb = ecg_pos_emb.to(device, non_blocking=True)
            with autocast(dtype=torch.float16):
                ecg_emb = model(ecg, ecg_pos_emb)
            embeddgins.extend(list(ecg_emb.cpu()))
            targets.extend(list(labels))
    return torch.stack(embeddgins).numpy(), torch.stack(targets).numpy()


def knn_online_eval(model, cfg):
    print("-------------------- KNN Online Evaluation --------------------")
    device = torch.device(cfg.device)
    train_loader, val_loader = create_loaders(cfg)
    X_train, y_train = collect_embeddings_and_targets(
        train_loader, model, device, set="train"
    )
    X_val, y_val = collect_embeddings_and_targets(val_loader, model, device, set="val")

    # 1. Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # you can change k
    knn.fit(X_train, y_train)

    # 2. Evaluate on validation set
    y_pred = knn.predict(X_val)
    y_prob = knn.predict_proba(X_val)[:, 1]  # probability for positive class

    # 3. Compute binary classification metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    knn_metrics = {
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-score": round(f1, 3),
        "AUC": round(auc, 3),
    }
    return knn_metrics
