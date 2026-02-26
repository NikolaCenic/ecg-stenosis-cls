from typing import Any, Tuple, Dict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import math
from collections import Counter
import wandb
import util.augmentations as augmentations


def log_set_distribution(dataset: Dataset, split: str):
    wandb.log({f"Set/{split}_total": len(dataset)})
    if dataset.positive_samples_cnt.ndim == 0:
        wandb.log(
            {
                f"Set/{split}_positive": dataset.positive_samples_cnt,
                f"Set/{split}_negative": dataset.negative_samples_cnt,
            }
        )
    else:
        # multilabel
        for ind in range(len(dataset.positive_samples_cnt)):
            wandb.log(
                {
                    f"Set/{split}_positive": dataset.positive_samples_cnt[ind],
                    f"Set/{split}_negative": dataset.negative_samples_cnt[ind],
                }
            )


class TimeSeriesDataset(Dataset):
    """
    Dataset for multi-domain time series analysis.
    """

    def __init__(
        self,
        data_path: str,
        labels_path: str = None,
        labels_mask_path: str = None,
        downstream_task: str = None,
        weighted_loss: bool = False,
        domain_offsets: Dict = None,
        domain_agnostic: str = False,
        univariate: bool = False,
        num_samples: int = -1,
        dataset_percentage: int = 100,
        train: bool = False,
        test: bool = False,
        N_val: int = 1,
        multilabel: bool = False,
        severity: int = 70,
        seconds: int = 10,
        fs: int = 500,
        args=None,
        augmentation=True,
    ) -> None:
        """
        labels_path: path to labels (finetuning / online evaluation)
        labels_mask_path: path to labels masks (finetuning / online evaluation)
        downstream_task: downstream task (finetuning / online evaluation)

        domain_offsets: offsets for pos_embed_y
        domain_agnostic: share pos_embed_y across domains

        univariate: analyse each variate independently
                    note - univariate analysis is domain agnostic

        train: training or validation / test
        N_val: nb of chunks to validate / test
        """
        # data defined as list with tuples (modality:str, sample:torch.tensor())
        data = torch.load(
            data_path, map_location=torch.device("cpu"), weights_only=False
        )  # load to ram
        if data[0][1].dim() == 3:
            domain = [
                (sample[0], sample[1].shape) for sample in data
            ]  # domain_shape : (C, V, T)
            data = [
                sample[1][None, ...] for sample in data
            ]  # data_shape : (1, C, V, T)
        else:
            # [None, ] to add auxiliary channel (C) dimension (similar to rgb in imgs)
            domain = [
                (sample[0], sample[1][None, ...].shape) for sample in data
            ]  # domain_shape : (C, V, T)
            data = [
                sample[1][None, None, ...] for sample in data
            ]  # data_shape : (1, C, V, T)
        data = [s[..., : fs * seconds] for s in data]
        self.univariate = univariate
        self.domain_agnostic = True if self.univariate else domain_agnostic

        self.dataset_percentage = dataset_percentage
        self.domain = domain
        self.domains = {
            domain: shape for domain, shape in sorted(list(set(self.domain)))
        }  # unique domains
        self.augmentation = augmentation
        self.fs = fs
        self.multilabel = multilabel

        domain_list = [mod[0] for mod in domain]
        unique_domains = list(set(domain_list))

        self.domain_weights = {}
        domain_counts = Counter(domain_list)
        total_domains = len(domain_list)
        num_unique_domains = len(unique_domains)
        for mod_current in unique_domains:
            count = domain_counts[mod_current]
            mod_weight = math.sqrt(total_domains / (num_unique_domains * count))
            self.domain_weights.update(
                {mod_current: torch.tensor(mod_weight, dtype=torch.float32)}
            )
        self.offsets = {}
        if domain_offsets is None:
            offset = 0
            for domain, shape in self.domains.items():
                self.offsets.update({domain: offset})
                if not self.domain_agnostic:
                    offset += shape[-2]
        else:
            self.offsets = domain_offsets
        self.data = data
        # one-hot-encdoded labels defined as torch.tensor()
        if labels_path:
            self.labels = torch.load(
                labels_path, map_location=torch.device("cpu"), weights_only=False
            )  # load to ram
        else:
            self.labels = torch.zeros(size=(len(self.data),))
        if len(self.labels.shape) == 1 and not multilabel:
            labels = []
            samples_to_remove = []
            for ind, l in enumerate(self.labels):
                if isinstance(severity, tuple):
                    if severity[0] < l < severity[1]:
                        samples_to_remove.append(ind)
                    else:
                        labels.append(list(np.eye(2)[l >= severity[1]]))
                else:
                    labels.append(list(np.eye(2)[l >= severity]))
            self.labels = torch.tensor(labels)
            self.data = [
                s for ind, s in enumerate(self.data) if ind not in samples_to_remove
            ]
        self.num_samples = (
            int(len(self.data) * self.dataset_percentage / 100)
            if num_samples == -1
            else num_samples
        )
        self.data = self.data[: self.num_samples]
        self.labels = self.labels[: self.num_samples]
        if multilabel:
            self.positive_samples_cnt = self.labels.sum(0)
            self.negative_samples_cnt = self.labels - self.positive_samples_cnt
        else:
            self.positive_samples_cnt = self.labels[:, 1].sum()
            self.negative_samples_cnt = self.labels[:, 0].sum()

        if labels_mask_path:
            self.labels_mask = torch.load(
                labels_mask_path, map_location=torch.device("cpu"), weights_only=False
            )  # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)
        self.nb_classes = self.labels.shape[1]
        self.downstream_task = downstream_task

        self.weighted_loss = weighted_loss
        self.class_weights = None
        if self.downstream_task == "classification" and weighted_loss == True:
            if multilabel:
                self.class_weights = torch.sqrt(
                    len(self.labels) / self.positive_samples_cnt
                )
            else:
                self.class_weights = torch.sqrt(
                    len(self.labels)
                    / self.labels.argmax(-1).unique(return_counts=True)[-1]
                )
            print(f"Class weights: {self.class_weights}")
        self.train = train
        self.test = test
        self.N_val = N_val
        self.args = args
        self._set_transform_()

    def _set_transform_(self):
        if self.augmentation == "custom":
            print("Custom augmetation set!")
            self.transform = transforms.Compose(
                [
                    augmentations.CropResizing(
                        fixed_resize_len=self.args.time_steps,
                        lower_bnd=self.args.crop_lower_bnd,
                        upper_bnd=self.args.crop_upper_bnd,
                        resize=True,
                    ),
                    augmentations.FTSurrogate(
                        phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5
                    ),
                    augmentations.Jitter(sigma=self.args.jitter_sigma),
                    augmentations.Rescaling(sigma=self.args.rescaling_sigma),
                    augmentations.FinetuneECGTransforms(fs=self.fs),
                ]
            )
        elif self.augmentation == "default":
            print("Default augmetation set!")
            self.transform = transforms.Compose(
                [
                    augmentations.CropResizing(
                        fixed_resize_len=self.args.time_steps,
                        lower_bnd=self.args.crop_lower_bnd,
                        upper_bnd=self.args.crop_upper_bnd,
                        resize=True,
                    ),
                    augmentations.FTSurrogate(
                        phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5
                    ),
                    augmentations.Jitter(sigma=self.args.jitter_sigma),
                    augmentations.Rescaling(sigma=self.args.rescaling_sigma),
                ]
            )
        else:
            self.transform = lambda x: x
            print("No augmetation!")

    def set_domain_offsets(self, domain_offsets: Dict = None):
        """set predefined domain offsets"""
        self.offsets = domain_offsets

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        # (1, C, V, T)
        data = self.data[idx]
        if self.univariate:
            # transform a multivariate time series with V variates into V univariate time series
            # (V, C, 1, T)
            data = data.permute(2, 1, 0, 3)

        # number of samples to process
        N = data.shape[0]

        # (N=N_val,   C, V, T)   if multivariate analysis,
        # (N=N_val*V, C, 1, T) else
        if self.train == False or self.augmentation == False:
            # validate / test on more than one chunk per sample
            # Calculate number of overlapping chunks
            stride = 32  # ili : 4, traffic : 32
            if self.test == True:
                stride = 1

            N_val = max(int(((data.shape[-1] - self.args.time_steps) / stride) + 1), 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            # Create overlapping chunks
            data_chunks = [
                data[..., i * stride : (i * stride + self.args.time_steps)]
                for i in range(N_val)
            ]

            # Concatenate chunks
            # (N, C, x, T), where x either V or 1
            data = torch.cat(data_chunks, dim=0)
        else:
            N_val = max(data.shape[-1] - self.args.time_steps + 1, 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            # Create random chunks
            data_chunks = [self.transform(data) for i in range(N_val)]

            # Concatenate chunks
            # (N, C, x, T), where x either V or 1
            data = torch.cat(data_chunks, dim=0)

        if self.downstream_task == "regression":
            label = self.labels[idx][..., self.args.lower_bnd : self.args.upper_bnd]
            label_mask = self.labels_mask[idx][
                ..., self.args.lower_bnd : self.args.upper_bnd
            ]
        else:
            label = self.labels[idx].type(torch.LongTensor)
            if not self.multilabel:
                label = label.argmax(dim=-1)
            label_mask = torch.ones_like(label)

        domain, _ = self.domain[idx]
        domain_offset = self.offsets[domain]

        label = [label for i in range(N)]
        label_mask = [label_mask for i in range(N)]
        domain = [domain for i in range(N)]

        return (
            data,
            label,
            label_mask,
            self.args.patch_size,
            domain_offset,
            domain,
            self.args.time_steps,
            self.univariate,
        )

    @staticmethod
    def collate_fn_ft(batch):
        # (B, C, V, T)
        data = torch.cat(
            [data.unsqueeze(0) for sample in batch for data in sample[0]], dim=0
        )
        # (B, 1) or (B, N)
        label = torch.stack([label for sample in batch for label in sample[1]], dim=0)

        if len(label.shape) > 1 and label.shape[-1] > 1:
            # for multilabel we need float targets
            label = label.float()
        # (B, 1)
        label_mask = torch.stack(
            [label_mask for sample in batch for label_mask in sample[2]], dim=0
        )

        grid_width = batch[0][0].shape[-1] // batch[0][3][-1]
        grid_height = batch[0][0].shape[-2] // batch[0][3][-2]
        # (B, V', T'), V'=V/p and T'=T/q
        pos_embed_y = (
            torch.arange(grid_height).view(-1, 1).repeat(len(batch), 1, grid_width)
            + 1
            + batch[0][4]
        )

        pos_embed_y = torch.stack(
            [
                pos_embed_y[idx]
                for idx, sample in enumerate(batch)
                for i in range(len(sample[0]))
            ],
            dim=0,
        )

        return data, label, label_mask, torch.LongTensor(pos_embed_y)
