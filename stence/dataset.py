from torch.utils.data import Dataset
import pandas as pd
import torch
import wfdb
import random
from tqdm import tqdm
from stenosis_classification.static_classification.mil_datasets import MILDataset
from stenosis_classification.static_classification.image_datasets import Transform
from stenosis_classification.static_classification.constants import (
    MAJOR_SEGMENTS,
    ARTERY_SEGMENTS,
)
import torch.nn.functional as F
from omegaconf import OmegaConf
import os
import omegaconf
from ast import literal_eval
from stenosis_classification.clip.augmentations import ECGTransforms
import json
import numpy as np

DATA_FILES = {
    "train": "../data_files/data_splits/train.csv",
    "val": "../data_files/data_splits/val.csv",
    "test": "../data_files/data_splits/test.csv",
}
ECG_ANGIO_MATCH = "../data_files/ecg_angio_match.csv"


class EcgAngioDataset(Dataset):
    def __init__(
        self,
        setting,
        use_patch_embeddings=False,  # used for mil-clip training
        ecg_length_seconds=10,
        multiframe=None,
        ecg_patch_size=[1, 24],
        ecg_pos_emb_offset=0,
        precomputed_angio_embeddings: bool | str = False,
        angio_augmentation=False,
        ecg_augmentation=None,
        num_samples=-1,
        drop_views=False,
        segments=MAJOR_SEGMENTS,
        severity=70,
        resolution=518,
    ):
        super().__init__()
        self.ecg_length_seconds = ecg_length_seconds
        self.use_patch_embeddings = use_patch_embeddings
        self.num_samples = num_samples
        self.setting = setting
        self.precomputed_angio_embeddings = precomputed_angio_embeddings
        self.ecg_patch_size = ecg_patch_size
        self.ecg_pos_emb_offset = ecg_pos_emb_offset
        self.segments = segments
        self.severity = severity
        if multiframe is None:
            multiframe = OmegaConf.create(
                {
                    "frames_per_view": 1,
                    "left": 0,
                    "right": 0,
                    "spacing_left": 0,
                    "spacing_right": 0,
                }
            )
        self.multiframe = multiframe
        self.angio_image_transform = Transform(
            backbone_type="vit",
            setting=setting,
            resolution=resolution,
            augmentation=angio_augmentation,
            frames_per_view=self.multiframe.frames_per_view,
            adjust_resolution=False,
        )
        if isinstance(drop_views, omegaconf.listconfig.ListConfig):
            drop_views = tuple(drop_views)
        self.drop_views = drop_views
        self.ecg_transform = ECGTransforms(
            augmentation=ecg_augmentation, setting=setting
        )

        self.load_samples()

    def get_patient_level_labels(self, annotation):
        segment_labels = [annotation[s] for s in self.segments]
        rca_label = max(
            [annotation[s] for s in ARTERY_SEGMENTS["RCA"] if s in self.segments]
        )
        lca_label = max(
            [annotation[s] for s in ARTERY_SEGMENTS["LCA"] if s in self.segments]
        )
        patient_label = max(segment_labels)

        return [
            s >= self.severity
            for s in [patient_label, rca_label, lca_label, *segment_labels]
        ]

    def load_samples(self):
        data = pd.read_csv(DATA_FILES[self.setting])
        data["shape"] = data["shape"].apply(literal_eval)
        ecg_angio_match = pd.read_csv(ECG_ANGIO_MATCH)
        data = data.merge(ecg_angio_match, on="angio_key")
        self.samples = []
        if self.precomputed_angio_embeddings:
            with open(
                os.path.join(
                    "precomputed_embeddings",
                    "embeddings",
                    self.precomputed_angio_embeddings,
                    "shapes.json",
                ),
                "r",
            ) as f:
                shapes = json.load(f)
        for (angio_key, ecg_file), angio_data in tqdm(
            data.groupby(["angio_key", "ecg_file"]), desc=f"Loading {self.setting} data"
        ):

            stenosis_targets = self.get_patient_level_labels(
                {
                    annot["ID_SEGMENT"]: annot["STENOSIS_DEGREE"]
                    for annot in literal_eval(angio_data.iloc[0]["annotation"])
                }
            )

            sample = {
                "angio_key": angio_key,
                "ecg_file": ecg_file,
                "target": stenosis_targets,
            }
            if self.precomputed_angio_embeddings:
                precomputed_embeddings_dir = os.path.join(
                    "precomputed_embeddings",
                    "embeddings",
                    str(self.precomputed_angio_embeddings),
                    self.setting,
                )
                sample["precomputed_angio_embeddings"] = []
                sample["shape"] = shapes[angio_key]
                for k in os.listdir(precomputed_embeddings_dir):
                    embedding_file = os.path.join(
                        precomputed_embeddings_dir,
                        k,
                        f"{angio_key}.dat",
                    )
                    if os.path.isfile(embedding_file):
                        sample["precomputed_angio_embeddings"].append(embedding_file)

            else:

                sample["angio_views"] = []
                for _, row in angio_data.iterrows():
                    angulation = MILDataset.get_angulation(row["metadata_path"])
                    if angulation == "not_present":
                        continue

                    sample["angio_views"].append(
                        {
                            "shape": row["shape"],
                            "key_frame": row["key_frame"],
                            "mmap_path": row["mmap_path"],
                            "angulation": angulation,
                        }
                    )
            self.samples.append(sample)

        print(f"Angio-ecg matches {self.setting} set: {len(self.samples)}")

    def load_ecg(self, sample):
        ecg, metadata = wfdb.rdsamp(sample["ecg_file"])
        ecg = torch.tensor(ecg).T.float()
        return self.ecg_transform(ecg, metadata["fs"], self.ecg_length_seconds)

    def load_angio(self, sample):
        if self.precomputed_angio_embeddings:
            transformation_cnt = random.randint(
                0, len(sample["precomputed_angio_embeddings"]) - 1
            )
            angio = torch.tensor(
                np.memmap(
                    sample["precomputed_angio_embeddings"][transformation_cnt],
                    dtype="float32",
                    mode="r",
                    shape=tuple(sample["shape"]),
                )
            )
            angulations = None

        else:
            angio = torch.stack(
                [
                    MILDataset.load_view_data(
                        v, self.angio_image_transform, self.multiframe
                    )
                    for v in sample["angio_views"]
                ]
            )
            angulations = torch.stack(
                [torch.tensor(v["angulation"]) for v in sample["angio_views"]]
            )

        patch_embeddings = (
            self.precomputed_angio_embeddings and self.use_patch_embeddings
        )
        use_images = not self.precomputed_angio_embeddings
        if (
            self.drop_views
            and self.setting == "train"
            and (patch_embeddings or use_images)
        ):
            num_views = len(angio)
            if isinstance(self.drop_views, tuple):
                k = int(num_views * (1 - random.uniform(*self.drop_views)))

            else:
                k = int(num_views * (1 - self.drop_views))
            k = max(1, k)
            views_to_keep = random.sample(list(range(num_views)), k)

            angio = angio[views_to_keep]
            if angulations is not None:
                angulations = angulations[views_to_keep]

        return angio, angulations

    def __len__(self):
        return self.num_samples if self.num_samples != -1 else len(self.samples)

    def __getitem__(self, ind):
        sample = self.samples[ind]
        ecg = self.load_ecg(sample)
        angio, angulations = self.load_angio(sample)

        return (
            ecg,
            self.ecg_patch_size,
            self.ecg_pos_emb_offset,
            angio,
            angulations,
            sample["angio_key"],
            sample["target"],
        )

    @staticmethod
    def pad_data_(x: torch.tensor, max_views):
        # x is either K, FPV, 3, H,W or 11, DIM (if precomputed embeddings)
        if x is None:
            return torch.empty(0, dtype=torch.float32)

        if x.shape[0] == max_views:
            # this is the case when the angio features are precomputed
            return x
        else:
            pad = torch.zeros((max_views - x.shape[0], *x.shape[1:]))
            return torch.cat([x, pad], axis=0)

    @staticmethod
    def collate_fn_(batch):
        ecg_data = torch.stack([s[0].unsqueeze(0) for s in batch]).float()

        # ecg pos embeddings
        ecg_patch_size = batch[0][1]
        ecg_pos_emb_offset = batch[0][2]
        grid_width = batch[0][0].shape[-1] // ecg_patch_size[-1]
        grid_height = batch[0][0].shape[-2] // ecg_patch_size[-2]
        pos_embed_y = (
            torch.arange(grid_height).view(-1, 1).repeat(len(batch), 1, grid_width)
            + 1
            + ecg_pos_emb_offset
        )
        # angio data
        angio_keys = [s[-2] for s in batch]
        stenosis_targets = torch.tensor([s[-1] for s in batch]).float()

        num_views_per_patient = [s[3].shape[0] for s in batch]
        angio_data = torch.stack(
            [EcgAngioDataset.pad_data_(s[3], max(num_views_per_patient)) for s in batch]
        ).float()

        angulations = torch.stack(
            [EcgAngioDataset.pad_data_(s[4], max(num_views_per_patient)) for s in batch]
        ).float()

        return (
            ecg_data,
            torch.LongTensor(pos_embed_y),
            angio_data,
            num_views_per_patient,
            angulations,
            angio_keys,
            stenosis_targets,
        )
