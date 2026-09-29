"""Card attribute classifier: an ImageNet-pretrained backbone with four 3-way
heads (number, color, shape, shading) on straightened 160x256 card crops
(see crops.py). Requires the `train` dependency group (torch/torchvision).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .synth.cards import COLORS, NUMBERS, SHADINGS, SHAPES

ATTRS: list[tuple[str, tuple]] = [("number", NUMBERS), ("color", COLORS), ("shape", SHAPES), ("shading", SHADINGS)]
MEAN = (0.485, 0.456, 0.406)  # ImageNet, RGB
STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "efficientnet_b0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "mobilenet_v3_large": (torchvision.models.mobilenet_v3_large, torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, 960),
}


class CardClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        ctor, weights, feat = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.backbone_name = backbone
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(feat, 3 * len(ATTRS)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) normalized RGB -> (B, 4, 3) logits, heads in ATTRS order."""
        return self.head(self.pool(self.features(x))).view(-1, len(ATTRS), 3)


def preprocess(crops_bgr: list[np.ndarray], device: str | torch.device = "cpu") -> torch.Tensor:
    """BGR uint8 crops (from crops.warp_card) -> normalized RGB batch."""
    x = torch.from_numpy(np.stack([c[..., ::-1] for c in crops_bgr]).copy()).permute(0, 3, 1, 2)
    x = x.to(device, torch.float32) / 255.0
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def decode(logits: torch.Tensor) -> list[dict]:
    """(B, 4, 3) logits -> per card {attr: value, attr_p: probability}."""
    probs = logits.float().softmax(-1).cpu().numpy()
    out = []
    for p in probs:
        card = {}
        for (name, values), pa in zip(ATTRS, p):
            i = int(pa.argmax())
            card[name] = values[i]
            card[f"{name}_p"] = float(pa[i])
        out.append(card)
    return out


def load(path: str, device: str = "cpu") -> CardClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CardClassifier(ckpt["backbone"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
