from transformers import ViTMAEModel
import yaml
import torch
from timm.models.layers import trunc_normal_
from stenosis_classification.otis.ecg_founder.finetune_model import ft_12lead_ECGFounder
from stenosis_classification.otis.xecg.xECG import xECG
from stenosis_classification.otis.echoing_ecg.src.model.echoingecg_model import (
    EchoingECG,
)

import torch.nn as nn


def load_ptacl_encoder():
    """
    https://huggingface.co/alsalivan/vitmae_ecg
    """
    model = ViTMAEModel.from_pretrained("alsalivan/vitmae_ecg")
    return model


def load_xecg():
    """
    https://huggingface.co/riccardolunelli/xECG_base_model_v1
    """

    model = xECG.from_pretrained("riccardolunelli/xECG_base_model_v1")
    return model


def load_echoing_ecg_encoder():
    """
    https://huggingface.co/mcintoshML/EchoingECG
    """

    # Load model config
    with open("echoing_ecg/src/configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    model = EchoingECG(model_cfg)
    model_weights = torch.load(
        "echoing_ecg/echoingecg.pt", weights_only=True, map_location="cpu"
    )
    model.load_state_dict(model_weights)
    return model.ecg_encoder


def load_ecgfounder(nb_classes, linear_probe):
    """
    https://huggingface.co/PKUDigitalHealth/ECGFounder
    """
    pth = "ecg_founder/12_lead_ECGFounder.pth"
    model = ft_12lead_ECGFounder(
        torch.device("cuda"), pth, nb_classes, linear_prob=linear_probe
    )
    return model


class Baseline(torch.nn.Module):
    def __init__(
        self,
        model: str,
        lin_probe: bool,
        nb_classes: int = 2,
        head_layers: int = 1,
        pretrained_weights: str = None,
    ):
        super().__init__()
        assert model.startswith("baseline_")

        self.model_type = model.split("_")[1]
        print(f"Loading baseline {self.model_type}!")
        if self.model_type == "xecg":
            self.backbone = load_xecg()
            in_features = self.backbone.embedding_size

        elif self.model_type == "echoing-ecg":
            self.backbone = load_echoing_ecg_encoder()
            in_features = self.backbone.embed_size

        elif self.model_type == "ptacl":
            self.backbone = load_ptacl_encoder()
            in_features = self.backbone.config.hidden_size

        elif self.model_type == "ecgfounder":
            self.backbone = load_ecgfounder(nb_classes=nb_classes, linear_probe=False)
            in_features = self.backbone.dense.in_features
            self.backbone.dense = nn.Identity()

        self.head = self.create_head(
            in_features=in_features, nb_classes=nb_classes, head_layers=head_layers
        )

        if lin_probe:
            self.freeze_encoder()
        if pretrained_weights is not None and pretrained_weights != "":
            self.load_state_dict(torch.load(pretrained_weights))
            print(f"Baseline weights loaded from {pretrained_weights}")

    def freeze_encoder(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        for name, param in self.head.named_parameters():
            param.requires_grad = False

    def create_head(self, in_features, nb_classes, head_layers):
        layers = [torch.nn.BatchNorm1d(in_features, affine=False, eps=1e-6)]
        for _ in range(head_layers - 1):
            layers.extend([nn.Linear(in_features, in_features), nn.ReLU()])
        layers.append(nn.Linear(in_features, nb_classes))
        head = nn.Sequential(*layers)

        for layer in head:
            if isinstance(layer, nn.Linear):
                trunc_normal_(layer.weight, std=0.01)
        return head

    def forward(self, x, pos_emb=None):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        if self.model_type == "xecg":
            # B, TS, LEADS
            x = x.moveaxis(1, 2)
            cls = self.backbone(x)[0]
        if self.model_type == "echoing-ecg":
            # B, LEADS, TS
            cls = self.backbone(x)["mean"]
        if self.model_type == "ptacl":
            # B, 1, LEADS, 2500
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            emb = self.backbone(x).last_hidden_state
            assert emb.shape[1] == 601
            cls = emb[:, 0]
        if self.model_type == "ecgfounder":
            # B,LEADS, 1500
            cls = self.backbone(x)
        return self.head(cls)
