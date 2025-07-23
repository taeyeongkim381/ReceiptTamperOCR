import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds to make experiments reproducible.

    Args:
        seed (int): Seed value to set for all relevant libraries.
    """
    print(f"Setting seed: {seed}")

    # Python built-in random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (CUDA)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variable for Python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)
    