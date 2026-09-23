import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducible experiments.

    Args:
        seed:
            Random seed value.

        deterministic:
            If True, request deterministic CUDA algorithms.
            This can reduce training performance.
    """

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Optional deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True