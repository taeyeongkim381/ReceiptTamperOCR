import torch
import torch.nn as nn
import torchvision.models as models
from transformers import RobertaForSequenceClassification

# -------------------------------
# 📌 ResNet50 이진 분류기
# -------------------------------
class ResNet50Binary(nn.Module):
    def __init__(self, pretrained=True, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)  # [B,2]

# -------------------------------
# 📌 RoBERTa-base 이진 분류기
# -------------------------------
class RobertaBinary(nn.Module):
    def __init__(self, pretrained_model_name="roberta-base", num_classes=2):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        # HuggingFace 모델은 dict 형태로 반환
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )


def get_model(model_type: str):
    """
    model_type: "resnet50" 또는 "roberta"
    """
    if model_type.lower() == "resnet50":
        return ResNet50Binary(pretrained=True)
    elif model_type.lower() == "roberta":
        return RobertaBinary(pretrained_model_name="roberta-base")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    