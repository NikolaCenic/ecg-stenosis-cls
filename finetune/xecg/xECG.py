import torch
from torch import nn

from xlstm import (
    FeedForwardConfig,
    mLSTMLayerConfig,
    mLSTMBlockConfig,
    sLSTMLayerConfig,
    sLSTMBlockConfig,
    xLSTMBlockStackConfig,
    xLSTMBlockStack,
)
import numpy as np
from huggingface_hub import PyTorchModelHubMixin


class xECG(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://github.com/dlaskalab/bench-xecg/",
    pipeline_tag="other",
    license="mit",
):
    """
    This is a simplified version of the xECG model, where most of the hyperparameters are hardcoded.
    This class should be used for the pretrained model on huggingface.
    """

    def __init__(
        self,
        cls_type,
        config,
    ):
        super(xECG, self).__init__()

        self.dropout = nn.Dropout(config["dropout"])
        self.sampling_freq = config["sampling_freq"]
        self.patch_size = config["patch_size"]
        self.embedding_size = config["embedding_size"]
        self.cls_type = cls_type
        assert self.cls_type in [
            "max",
            "avg",
            "mean",
            None,
        ], f"cls_type {self.cls_type} not supported"

        self.patch_embedding = LinearPatchEmbedding(
            patch_size=config["patch_size"],
            num_hiddens=config["embedding_size"],
            num_channels=12,
        )

        self.core = get_xlstm(config)
        self.mask_token = nn.Parameter(torch.zeros(config["embedding_size"]))

    def pooling(self, out, padding_mask=None):
        cls = None
        if self.cls_type == "max":
            if padding_mask is None:
                cls = out.max(dim=1)[0]
            else:
                # do not consider padded value in max
                cls = out.masked_fill(padding_mask, -torch.inf).max(dim=1)[0]
        elif self.cls_type == "mean" or self.cls_type == "avg":
            if padding_mask is None:
                cls = out.mean(dim=1)
            else:
                # do not consider padded value in mean
                cls = out.masked_fill(padding_mask, 0).sum(dim=1) / (
                    out.shape[1] - padding_mask.sum(dim=1)
                ).clamp(min=1)
        return cls, out

    def forward(self, x):
        # find the padded part of the signal
        padding_mask = self.get_padding_mask(x)

        x = self.patch_embedding(x)

        out = self.core(x)  # [batch_size, embedding_dim]
        cls, out = self.pooling(out, padding_mask)

        return cls, out

    def get_padding_mask(self, x):
        padding_mask = (x.abs().sum(dim=-1) == 0).unsqueeze(-1)
        num_patches = x.shape[1] // self.patch_size
        padding_mask_patched = (
            padding_mask.view(-1, num_patches, self.patch_size)[:, :, 0]
            .unsqueeze(-1)
            .expand(-1, -1, self.embedding_size)
        )
        return padding_mask_patched

    def trainable_parameters(self):
        return self.parameters()

    def get_layers(self):
        """
        This function should return the layers of the model where to apply the layerwise decay
        """
        return self.core.model.blocks

    def additional_params(self, lr, last_layer_lr, wd):
        """
        This fucntion should return additional parameters used by a model (like classification token and so on...)
        """
        params = []
        params.append(
            {
                "params": self.patch_embedding.parameters(),
                "lr": last_layer_lr,
                "name": "patch_embedding",
                "weight_decay": wd,
            }
        )

        if hasattr(self.core, "post_blocks_norm"):
            params.append(
                {
                    "params": self.core.post_blocks_norm,
                    "lr": lr,
                    "name": "post_block_norm",
                    "weight_decay": wd,
                }
            )

        return params

    def format_keys(self, key):
        if key.startswith("model."):
            key = key[6:]

        key = key.replace(
            "xlstm.model", "core.model"
        )  # Remove 'module.' prefix if present
        return key

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        new_state_dict = {
            self.format_keys(k): v for k, v in checkpoint["state_dict"].items()
        }

        # for k, v in new_state_dict.items():
        #    if "slstm_cell._recurrent_kernel_" in k:
        #        new_state_dict[k] = v.permute(0, 2, 1)

        # remove the fc layer
        new_state_dict = {k: v for k, v in new_state_dict.items() if "fc" not in k}
        message = self.load_state_dict(new_state_dict, strict=False)
        print(message)


class LinearPatchEmbedding(nn.Module):
    def __init__(self, patch_size=64, num_hiddens=256, num_channels=12):
        super().__init__()
        self.conv = nn.Conv1d(
            num_channels,
            num_hiddens,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x, permute=True):
        if permute:
            x = x.permute(0, 2, 1)  # put the channels in the middle
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x


class vanillaxLSTMWrapper(nn.Module):
    """xlstm wrapper to allow bidirectionality and drop path"""

    def __init__(self, xlstm, dropout=0.2, bidirectional=False, drop_path=0.0):
        super(vanillaxLSTMWrapper, self).__init__()
        self.model = xlstm
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.drop_path = DropPath()
        self.dropout_rates = [
            x.item() for x in torch.linspace(0, drop_path, len(self.model.blocks))
        ]

    def step(self, x, state=None):
        return self.model.step(x, state=state)

    def forward(self, x: torch.Tensor):

        for i, block in enumerate(self.model.blocks):
            if self.bidirectional:
                # flip the sequence
                if i > 0:
                    x = x.flip(1)

            if self.dropout_rates[i] == 0.0 or not self.training:
                x = block(x)
            else:
                x = self.drop_path(x, block, self.dropout_rates[i])

        x = self.model.post_blocks_norm(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in the main path of residual blocks)."""

    def __init__(self, is_large_mlstm=False):
        super(DropPath, self).__init__()
        self.is_large_mlstm = is_large_mlstm

    def forward(self, x, block, drop_path_prob, state=None):
        if drop_path_prob == 0.0 or not self.training:
            if self.is_large_mlstm:
                return block(x, state)
            else:
                return block(x)

        # indexes of the batch
        idxs = torch.randperm(x.shape[0])
        num_to_keep = int(np.ceil((1.0 - drop_path_prob) * x.shape[0]))
        idxs_to_keep = idxs[:num_to_keep]  # First N elements are kept

        if self.is_large_mlstm:
            out, _ = block(x[idxs_to_keep], None)
            x[idxs_to_keep] = out
            # dont need to have a state in training
            return x, None
        else:
            x[idxs_to_keep] = block(x[idxs_to_keep])
            return x


def get_xlstm(config):
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4,
                qkv_proj_blocksize=config["num_heads"],
                num_heads=config["num_heads"],
                proj_factor=config["proj_factor"],
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=config["num_heads"],
                backend=(
                    config["backend"]
                    if "backend" in config.keys() and config["backend"]
                    else "cuda"
                ),
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
                batch_size=config["batch_size"],
            ),
            feedforward=FeedForwardConfig(
                proj_factor=1.3, act_fn=config["activation_fn"]
            ),
        ),
        context_length=8000,
        num_blocks=len(config["xlstm_config"]),
        embedding_dim=config["embedding_size"],
        slstm_at=[idx for idx, b in enumerate(config["xlstm_config"]) if b == "s"],
        dropout=config["dropout"],
        add_post_blocks_norm=(
            config["use_final_layer_norm"]
            if "use_final_layer_norm" in config.keys()
            else False
        ),
    )
    print(
        "creating xlstm with slstm at: ",
        [idx for idx, b in enumerate(config["xlstm_config"]) if b == "s"],
    )

    return vanillaxLSTMWrapper(
        xLSTMBlockStack(cfg),
        dropout=config["dropout"],
        bidirectional=True,
        drop_path=config["drop_path_prob"],
    )


if __name__ == "__main__":
    model = xECG.from_pretrained("riccardolunelli/xECG_base_model_v1")
    print(model)
