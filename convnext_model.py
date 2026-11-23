import torch
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from lightning_modules.detector import DeepfakeDetector


def build_convnext_tiny(pretrained=False):
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    return model


def load_convnext_checkpoint(checkpoint_path):
    backbone = build_convnext_tiny()
    detector = DeepfakeDetector.load_from_checkpoint(
        checkpoint_path,
        model=backbone,
        lr=1e-4,
        map_location="cpu",
    )
    detector.model.eval()
    return detector.model
