from torch import nn as nn
from stenosis_classification.clip.utils import load_ecg_model, load_angio_model
from stenosis_classification.static_classification.models import SegmentMIL
from stenosis_classification.static_classification.constants import MAJOR_SEGMENTS
from typing import List
import torch
import yaml
from stenosis_classification.otis.echoing_ecg.src.model.echoingecg_model import (
    EchoingECG,
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000


def load_echoing_ecg():
    # Load model config
    with open("../finetune/echoing_ecg/src/configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)

    model = EchoingECG(model_cfg)
    model_weights = torch.load(
        "../finetune/echoing_ecg/echoingecg.pt", weights_only=True, map_location="cpu"
    )
    model.load_state_dict(model_weights)
    return model.ecg_encoder


class ECGModelWrapper(nn.Module):
    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.model_type = cfg.get("type", "otis")
        if self.model_type == "otis":
            self.model, self.ecg_pos_emb_offset = load_ecg_model(cfg)
            feature_emb = 192
        elif self.model_type == "echoing-ecg":

            self.model = load_echoing_ecg()
            self.ecg_pos_emb_offset = 5
            feature_emb = self.model.embed_size
        else:
            ValueError

        self.projector = nn.Sequential(
            nn.LayerNorm(feature_emb),
            nn.Linear(feature_emb, feature_emb),
            nn.ReLU(),
            nn.Linear(feature_emb, embed_dim),
        )
        self.projector.apply(init_weights)
        self.unfreeze()

    def unfreeze(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x, ecg_pos_emb):
        if self.model_type == "otis":
            features = self.model(x, ecg_pos_emb)
        if self.model_type == "echoing-ecg":
            x = x.squeeze(1)
            features = self.model(x)["mean"]
        else:
            raise ValueError()
        return self.projector(features)


class AngioModelWrapper(nn.Module):
    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.model = load_angio_model(cfg)
        self.cfg = cfg
        if cfg.precomputed_embeddings:
            feature_emb = 512
        else:
            feature_emb = self.model.embed_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(feature_emb),
            nn.Linear(feature_emb, feature_emb),
            nn.ReLU(),
            nn.Linear(feature_emb, embed_dim),
        )

        self.projector.apply(init_weights)

    def obtain_cls_token(self, features):
        selectors_dict = {
            "p": lambda x: x[:, 0][
                :,
                None,
            ],
            "r": lambda x: x[:, 1][
                :,
                None,
            ],
            "l": lambda x: x[:, 2][
                :,
                None,
            ],
            "a": lambda x: x[:, [1, 2]],
            "s": lambda x: x[:, 3:],
        }
        aggregators_dict = {
            "max": lambda x: x.max(dim=1).values,
            "mean": lambda x: x.mean(dim=1),
        }
        selector = self.cfg.cls_token_method.split("_")[0]
        features = torch.cat([selectors_dict[s](features) for s in selector], dim=1)

        if features.shape[1] > 1:
            aggregator = self.cfg.cls_token_method.split("_")[1]
            features = aggregators_dict[aggregator](features)
        return features.squeeze(1)

    def forward(self, x, samples_cnt, angulations):
        if not self.cfg.precomputed_embeddings:
            x, _ = self.model.forward_full_backbone(x, samples_cnt, angulations)

        return self.projector(self.obtain_cls_token(x))


class CLIPSegmentMIL(SegmentMIL):
    def __init__(self, clip_embed_dim, frames_per_view):
        feature_emb = 512
        super().__init__(
            backbone_type="vit",
            pretrained=True,
            encode_level="patch",
            embed_dim=feature_emb,
            resolution=518,
            projector_layers=0,
            classifier_layers=0,
            attention_type="transformer",
            attention_heads=4,
            shared_classifier=False,
            hierarchical=True,
            transformer_layers=2,
            segments_to_use=MAJOR_SEGMENTS,
            frames_per_view=frames_per_view,
            cnt_pos_embeddings=256,
            extra_queries=4,
        )
        del self.encoder
        del self.projector

        self.clip_projector = nn.Sequential(
            nn.LayerNorm(feature_emb),
            nn.Linear(feature_emb, feature_emb),
            nn.ReLU(),
            nn.Linear(feature_emb, clip_embed_dim),
        )
        self.clip_projector.apply(init_weights)

    def forward(
        self,
        patch_features: torch.tensor,
        samples_cnt: List[int],
    ):
        if len(patch_features.shape) == 4:
            patch_features = patch_features.unsqueeze(2)
        self.bs, self.max_samples_per_patient, _, self.tokens_per_frame, _ = (
            patch_features.shape
        )
        y, _ = self.forward_attention(
            self.query.repeat(self.bs, 1, 1), patch_features, samples_cnt
        )
        clip_embedding = self.clip_projector(y[:, 0])
        stenosis_logits = self.classifier(y[:, 1:]).squeeze()
        return clip_embedding, stenosis_logits


if __name__ == "__main__":
    frames_per_view = 3
    m = ECGModelWrapper({"type": "echoing-ecg"}, embed_dim=128).cuda()
    data = torch.rand(10, 1, 12, 500).cuda()
    m.eval()
    print(m(data, None).shape)
