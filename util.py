import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed: int = 42):
    """
    실험 재현성을 높이기 위해 시드를 고정하는 함수
    """
    print(f"🔧 Setting seed: {seed}")
    # 파이썬 자체 랜덤
    random.seed(seed)
    # Numpy 랜덤
    np.random.seed(seed)
    # PyTorch CPU 시드
    torch.manual_seed(seed)
    # PyTorch CUDA 시드
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 동작 고정 (deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 추가: 환경 변수로도 시드 고정
    os.environ["PYTHONHASHSEED"] = str(seed)


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class or binary classification.
    gamma: focusing parameter (e.g., 2.0)
    alpha: class weighting (list, tensor, or scalar). 
           If scalar, same weight applied to all classes.
           If list/tensor, should be of size [num_classes].
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C) logits
        targets: (N,) int64 class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = prob of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # gather alpha weight per sample
            if self.alpha.numel() == 1:
                alpha_t = self.alpha.to(inputs.device)
            else:
                alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
