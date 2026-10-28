"""2026 public-data reconstruction, not the missing 2025 company implementation."""
import torch
from torch import nn
import torch.nn.functional as F
from vendor.dformerv2_backbone import DFormerv2_S
from vendor.mlp_decoder import DecoderHead


class RGBTSegmenter(nn.Module):
    def __init__(self, variant="rgbt"):
        super().__init__()
        if variant not in ("rgbt", "rgb-only"):
            raise ValueError(f"Unknown input variant: {variant}")
        self.variant = variant
        self.backbone = DFormerv2_S(drop_path_rate=0.1)
        self.decode_head = DecoderHead(
            in_channels=[64, 128, 256, 512], num_classes=9, embed_dim=128,
            norm_layer=nn.BatchNorm2d,
        )

    def load_encoder(self, path):
        source = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("model", "state_dict"):
            if key in source:
                source = source[key]
        target = self.backbone.state_dict()
        selected = {}
        for key, value in source.items():
            key = key.removeprefix("module.").removeprefix("backbone.")
            if key in target and value.shape == target[key].shape:
                selected[key] = value
        result = self.backbone.load_state_dict(selected, strict=False)
        missing = [k for k in result.missing_keys if not k.endswith("num_batches_tracked")]
        if missing:
            raise RuntimeError(f"Encoder checkpoint is incomplete: {missing[:12]}")
        return {"loaded_tensors": len(selected), "missing_keys": result.missing_keys}

    def train(self, mode=True):
        super().train(mode)
        # Retain pretrained RGB normalization with the small single-GPU batches.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, rgb, thermal):
        # T is a normalized intensity map, not a metric depth estimate.
        # A constant map removes every auxiliary-intensity difference from the
        # prior. The RGB-only control retains the same spatial prior/architecture.
        if self.variant == "rgb-only":
            thermal = torch.full_like(thermal, 0.5)
        features = self.backbone(rgb, thermal)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
