# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : TimeSeriesImputation
# @File          : data_base.py
# @Time          : 2024/7/3 16:21
# @Author        : SY.M
# @Software      : PyCharm
import copy
from typing import Callable, Union

import torch

from data_processing.data_processing_uitls import create_missing, get_mask, padding
from data_processing.dataset_beijing_air_quality import preprocess_beijing_air_quality
from data_processing.dataset_electricity_load_diagrams import preprocess_electricity_load_diagrams
from data_processing.dataset_ett import preprocess_ett
from data_processing.dataset_italy_air_quality import preprocess_italy_air_quality
from data_processing.dataset_pedestrian import preprocess_ucr_uea_datasets
from data_processing.dataset_pems_traffic import preprocess_pems_traffic
from data_processing.dataset_physionet_2012 import preprocess_physionet2012
from data_processing.dataset_physionet_2019 import preprocess_physionet2019

# from benchpots.datasets import preprocess_ucr_uea_datasets, preprocess_beijing_air_quality

from torch.utils.data import Dataset, DataLoader


class Data_Base:
    def __init__(self,
                 BATCHSIZE: int,
                 BATCHSIZE_Test: int,
                 selected_dataset: str,
                 padding_mode: Union[int, float, str],
                 seq_len: int,
                 train_missing_ratio: float,
                 train_pattern: str,
                 test_missing_ratio: float,
                 test_pattern: str,
                 block_len_width: int,
                 Dataset_Class: Callable,
                 subseq_len: int,
                 block_mr: float,
                 DEVICE: torch.device = torch.device('cpu'),
                 scalar: bool = True,
                 shuffle: bool = True,
                 ):
        test_pattern = train_pattern if test_pattern is None else test_pattern
        test_missing_ratio = train_missing_ratio if test_missing_ratio is None else test_missing_ratio

        assert train_pattern in ['point', 'subseq', 'block'], (
            'pattern must be either "point" or "subseq" or "block" but got {}'.format(train_pattern))
        assert test_pattern in ['point', 'subseq', 'block'], (
            'pattern must be either "point" or "subseq" or "block" but got {}'.format(test_pattern))
        assert selected_dataset in ['Air', 'Electricity_Load', 'ETT', 'Physionet2012',
                                    'Physionet2019', 'Italy_Air', 'Pems_Traffic', 'Pedestrian'], \
            f'{selected_dataset} not supported!'

        dataset_info, train_data, val_data, test_data = processing_dataset(selected_dataset=selected_dataset,
                                                                           train_rate=train_missing_ratio,
                                                                           test_rate=test_missing_ratio,
                                                                           seq_len=seq_len,
                                                                           train_pattern=train_pattern,
                                                                           test_pattern=test_pattern,
                                                                           padding_mode=padding_mode,
                                                                           block_len_width=block_len_width,
                                                                           sub_seq_len=subseq_len,
                                                                           block_mr=block_mr,
                                                                            )

        self.train_dataset = Dataset_Class(data_dict=train_data, DEVICE=DEVICE)
        self.val_dataset = Dataset_Class(data_dict=val_data, DEVICE=DEVICE)
        self.test_dataset = Dataset_Class(data_dict=test_data, DEVICE=DEVICE)

        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=BATCHSIZE, shuffle=shuffle)
        self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=BATCHSIZE, shuffle=shuffle)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=BATCHSIZE_Test, shuffle=False)

        self.d_feature = self.train_dataset.d_feature
        self.seq_len = self.train_dataset.seq_len

        # input parameters
        self.BATCHSIZE = BATCHSIZE
        self.DEVICE = DEVICE
        self.selected_dataset = selected_dataset
        self.train_missing_ratio = train_missing_ratio
        self.test_missing_ratio = test_missing_ratio
        self.scalar = scalar

class Dataset_Base(Dataset):
    def __init__(self,
                 data_dict: dict,
                 DEVICE: torch.device = None):
        super(Dataset_Base, self).__init__()

        assert 'X_intact' in data_dict.keys(), f'The data_dict requires X_intact, but get {data_dict.keys()}'
        assert 'X' in data_dict.keys(), f'The data_dict requires X, but get {data_dict.keys()}'
        assert 'missing_mask' in data_dict.keys(), f'The data_dict requires missing_mask, but get {data_dict.keys()}'
        assert 'indicating_mask' in data_dict.keys(), f'The data_dict requires indicating_mask, but get {data_dict.keys()}'

        self.data_dict = data_dict
        self.sample_num = self.data_dict['X_intact'].shape[0]
        self.seq_len = self.data_dict['X_intact'].shape[1]
        self.d_feature = self.data_dict['X_intact'].shape[2]
        self.to_device(DEVICE)

    def to_device(self,
                  DEVICE: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        for item in self.data_dict.items():
            self.data_dict[item[0]] = item[1].to(DEVICE) if item[1] is not None else item[1]

    def getitem_core(self, item):

        temp: dict[str, Union[torch.Tensor, int]] = dict(
            sample_idx=item,
            X_intact=self.data_dict['X_intact'][item],
            X=self.data_dict['X'][item],
            missing_mask=self.data_dict['missing_mask'][item],
            indicating_mask=self.data_dict['indicating_mask'][item],)

        return temp

    def __getitem__(self, item):
        return self.getitem_core(item=item)

    def __len__(self):
        return self.sample_num


def processing_dataset(selected_dataset,
                       seq_len,
                       train_rate,
                       test_rate,
                       train_pattern,
                       test_pattern,
                       padding_mode,
                       **kwargs):

    assert 0 <= train_rate < 1, f"rate must be in [0, 1), but got {train_rate}"
    assert 0 <= test_rate < 1, f"rate must be in [0, 1), but got {test_rate}"
    assert seq_len > 0, f"sample_n_steps must be larger than 0, but got {seq_len}"

    PROCESS_DICT = dict(Air=preprocess_beijing_air_quality,
                        Electricity_Load=preprocess_electricity_load_diagrams,
                        ETT=preprocess_ett,
                        Physionet2012=preprocess_physionet2012,
                        Physionet2019=preprocess_physionet2019,
                        Italy_Air=preprocess_italy_air_quality,
                        Pems_Traffic=preprocess_pems_traffic,
                        Pedestrian=preprocess_ucr_uea_datasets
                        )

    data_processing_method = PROCESS_DICT[selected_dataset]

    dataset_info, train_data, val_data, test_data = data_processing_method(seq_len=seq_len,)

    train_X_intact, val_X_intact, test_X_intact = (copy.deepcopy(train_data['X_intact']),
                                                   copy.deepcopy(val_data['X_intact']),
                                                   copy.deepcopy(test_data['X_intact']))

    assert train_data['X_intact'] is not val_data['X_intact'], "train and val X_intact point to same object!"
    assert train_data['X_intact'] is not test_data['X_intact'], "train and test X_intact point to same object!"
    assert val_data['X_intact'] is not test_data['X_intact'], "val and test X_intact point to same object!"

    if test_rate > 0 or train_rate > 0:
        # Artificial missing masks are generated after normalization and are fixed for
        # train/val/test within this run. Models receive the same missing_mask and
        # indicating_mask tensors from this data-processing step.
        train_X, val_X, test_X = create_missing(train_rate=train_rate,
                                                test_rate=test_rate,
                                                train_X_intact=train_X_intact,
                                                val_X_intact=val_X_intact,
                                                test_X_intact=test_X_intact,
                                                train_pattern=train_pattern,
                                                test_pattern=test_pattern,
                                                **kwargs)

        train_data["X"] = train_X
        train_data["missing_mask"], train_data['indicating_mask'] = get_mask(X=train_X, X_intact=train_X_intact)

        val_data["X"] = val_X
        val_data["missing_mask"], val_data['indicating_mask'] = get_mask(X=val_X, X_intact=val_X_intact)

        test_data["X"] = test_X
        test_data["missing_mask"], test_data['indicating_mask'] = get_mask(X=test_X, X_intact=test_X_intact)

    train_data["X_intact"], train_data['X'] = padding(X=train_data["X"], X_intact=train_X_intact, nan=padding_mode)
    val_data["X_intact"], val_data['X'] = padding(X=val_data["X"], X_intact=val_X_intact, nan=padding_mode)
    test_data["X_intact"], test_data['X'] = padding(X=test_data["X"], X_intact=test_X_intact, nan=padding_mode)

    return dataset_info, train_data, val_data, test_data
