# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : TimeSeriesImputation
# @File          : random_seed.py
# @Time          : 2024/7/18 15:33
# @Author        : SY.M
# @Software      : PyCharm

import torch
import numpy as np


def setup_seed(seed):
    """
    Set random seeds.
    :param seed: random seed value
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
