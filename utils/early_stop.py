# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : TimeSeriesImputation
# @File          : early_stop.py
# @Time          : 2024/7/22 15:28
# @Author        : SY.M
# @Software      : PyCharm
from copy import deepcopy

import numpy as np
import torch.nn


class EarlyStop:
    def __init__(self,
                 patience: int = 10):
        self.patience = patience
        self.current_patience = patience
        self.best_result = np.inf
        self.best_model_dict = None

    def if_stop(self,
                result: float,
                net: torch.nn.Module) -> bool:
        if result < self.best_result:
            self.best_result = result
            net.eval()
            self.best_model_dict = deepcopy(net.state_dict())
            self.current_patience = self.patience
        else:
            self.current_patience -= 1

        if self.current_patience <= 0:
            return True
        else:
            return False


