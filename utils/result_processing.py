# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : TimeSeriesImputation
# @File          : result_processing_final.py
# @Time          : 2024/7/22 13:54
# @Author        : SY.M
# @Software      : PyCharm
import torch
import numpy as np
from pypots.nn.functional import calc_mae, calc_mre, calc_mse


class Result_base:
    def __init__(self):
        # Epoch-wise parameters
        self.X_intact = []
        self.X = []
        self.missing_mask = []
        self.indicating_mask = []
        self.output = []

    def record_epoch_data(self,
                          input: dict,
                          output: torch.Tensor
                          ):
        l = lambda x: x.cpu().detach().numpy()
        self.X_intact.append(l(input['X_intact']))
        self.X.append(l(input['X']))
        self.missing_mask.append(l(input['missing_mask']))
        self.indicating_mask.append(l(input['indicating_mask']))
        self.output.append(l(output))

    def cal_epoch_loss(self,
                       stage: str):
        try:
            assert stage in ["train", "val", "test"], f'Require stage in ["train", "val", "test"] but got {stage}'

            self.X_intact = np.concatenate(self.X_intact, axis=0)
            self.X = np.concatenate(self.X, axis=0)
            self.missing_mask = np.concatenate(self.missing_mask, axis=0)
            self.indicating_mask = np.concatenate(self.indicating_mask, axis=0)
            self.output = np.concatenate(self.output, axis=0)

            MIT_mae = calc_mae(self.output, self.X_intact, self.indicating_mask)
            MIT_mse = calc_mse(self.output, self.X_intact, self.indicating_mask)
            MIT_mre = calc_mre(self.output, self.X_intact, self.indicating_mask)

        finally:
            self.clear_epoch_data()

        return MIT_mae, MIT_mse, MIT_mre

    def clear_epoch_data(self):
        self.X_intact = []
        self.X = []
        self.missing_mask = []
        self.indicating_mask = []
        self.output = []

    def visualization(self):
        pass

    def print_result(self,
                     MIT_mae: float,
                     MIT_mse: float,
                     MIT_mre: float,
                     stage: str = 'train',
                     time_consume: float = None,
                     logger = None,
                     console: bool = False,
                     log_result: bool = True,
                     console_prefix: str = None,):

        assert stage in ['train', 'val', 'test'], f'Require stage in ["train", "val", "test"] but got {stage}'

        content = (
            f'{stage.upper()}: MAE:{round(MIT_mae, 4)}\t'
            f'MSE:{round(MIT_mse, 4)}\tMRE:{round(MIT_mre, 4)}'
        )
        if time_consume is not None:
            content += f'\tTime Consume:{round(time_consume, 4)}'

        if logger is not None and log_result:
            logger.info(content, pos='blank')
        elif logger is None and not console:
            print(content, flush=True)
        if console:
            print(f"{console_prefix} {content}" if console_prefix else content, flush=True)


