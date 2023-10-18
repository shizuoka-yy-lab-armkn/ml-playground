import random

import numpy as np
import torch


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all() は torch.manual_seed() 内部で呼ばれるので不要
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
