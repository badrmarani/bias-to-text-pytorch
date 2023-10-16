import random
from contextlib import contextmanager

import torch

import numpy as np


@contextmanager
def seed_everything(seed=42):
    random.seed(seed)
    np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
    yield
