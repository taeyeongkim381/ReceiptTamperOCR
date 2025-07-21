import random
import numpy as np
import torch
import os

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
