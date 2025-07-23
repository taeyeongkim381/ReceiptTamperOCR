import torch
import torch.nn as nn
import torchvision.models as models
from transformers import RobertaForSequenceClassification
from typing import Literal


class ResNet50Binary(nn.Module):
    """ResNet50 backbone for binary classification."""
    def __init__(self, pretrained: bool = True, num_classes: int = 2) -> None:
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, num_classes]


class RobertaBinary(nn.Module):
    """RoBERTa model for binary classification."""
    def __init__(self, pretrained_model_name: str = "roberta-base", num_classes: int = 2) -> None:
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


def get_model(model_type: Literal["resnet50", "roberta"]) -> nn.Module:
    """Return a model instance based on type."""
    if model_type.lower() == "resnet50":
        return ResNet50Binary(pretrained=True)
    elif model_type.lower() == "roberta":
        return RobertaBinary(pretrained_model_name="roberta-base")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    