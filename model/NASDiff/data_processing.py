# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : TimeSeriesImputation
# @File          : data_processing.py
# @Time          : 2024/9/18 16:46
# @Author        : SY.M
# @Software      : PyCharm


from data_processing.data_base import Dataset_Base
import torch


class DatasetFor_NASDiff(Dataset_Base):
    def __init__(self,
                 data_dict: dict,
                 DEVICE: torch.device = None):
        super().__init__(data_dict=data_dict,
                         DEVICE=DEVICE)
