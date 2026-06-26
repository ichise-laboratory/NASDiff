# python3.7
# -*- coding: utf-8 -*-
# @Project_Name  : TimeSeriesImputation
# @File          : data_processing_uitls.py
# @Time          : 2024/7/3 18:27
# @Author        : SY.M
# @Software      : PyCharm


from typing import Union, Tuple
import torchcde
import numpy as np
import pandas as pd
import torch
from einops import rearrange, repeat
from pygrinder import mcar, seq_missing, block_missing


def sliding_window(data: Union[np.ndarray, torch.Tensor],
                   window_size: int,
                   time_data: Union[np.ndarray, torch.Tensor] = None,
                   sliding_len: int = None):
    if time_data is not None:
        assert data.shape[0] == time_data.shape[0], (f"We need data and time_data have the same length at 1st "
                                                     f"dimension but got {data.shape[0]} and {time_data.shape[0]}")

    sliding_len = window_size if sliding_len is None else sliding_len
    total_len = data.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len
    if total_len - start_indices[-1] * sliding_len < window_size:  # remove the last one if left length is not enough
        start_indices = start_indices[:-1]
    sample_collector = []
    time_collector = []
    for idx in start_indices:
        sample_collector.append(data[idx: idx + window_size])
        time_collector.append(time_data[idx: idx + window_size]) if time_data is not None else None

    if time_data is not None:
        return np.asarray(sample_collector).astype('float32'), np.asarray(time_collector).astype('float32')
    else:
        return np.asarray(sample_collector).astype('float32')


def specify_type(data: [Union[np.ndarray, torch.Tensor, pd.DataFrame]],
                 specify_type: str = 'tensor') -> [Union[np.ndarray, torch.Tensor, pd.DataFrame]]:
    assert specify_type in ['tensor', 'numpy', 'dataframe'], \
        f'Supported data type includes tensor, numpy and dataframe, but got {specify_type}!'

    output = None
    if specify_type == 'numpy':
        if isinstance(data[0], np.ndarray):
            output = data
        elif isinstance(data[0], torch.Tensor):
            output = [i.numpy() for i in data]
        elif isinstance(data[0], pd.DataFrame):
            output = [i.to_numpy() for i in data]
    elif specify_type == 'tensor':
        if isinstance(data[0], torch.Tensor):
            output = [i.type(torch.float32) for i in data]
        elif isinstance(data[0], np.ndarray):
            output = [torch.from_numpy(i).type(torch.float32) for i in data]
        elif isinstance(data[0], pd.DataFrame):
            output = [torch.from_numpy(i.to_numpy()) for i in data]
        elif isinstance(data[0], float):
            output = [torch.Tensor([i]) for i in data]
        elif isinstance(data[0], int):
            output = [torch.Tensor([i]) for i in data]
    elif specify_type == 'dataframe':
        if isinstance(data[0], pd.DataFrame):
            output = data
        elif isinstance(data[0], np.ndarray):
            output = [pd.DataFrame(i) for i in data]
        elif isinstance(data[0], torch.Tensor):
            output = [pd.DataFrame(i) for i in data]

    if len(output) == 1:
        output = output[0]

    return output


def mcar_input(data: Union[np.ndarray, torch.Tensor],
               missing_ratio: float = 0.1,
               nan: float = 0.0, ) -> dict:
    assert 1 > missing_ratio > 0, f"Missing ratio must be between 0 and 1 but got {missing_ratio}!"

    data = specify_type([data, ], 'tensor')

    data = data.clone()  # clone X to ensure values of X out of this function not being affected

    # get data with artificial missing values (has original missing values, so after the mcar, the missing ratio is between [original missing, original missing + p])
    X = mcar(X=data, p=missing_ratio)

    X_intact = torch.clone(data)  # keep a copy of originally observed values in X_intact

    # create indicator matrix
    indicating_mask = ((~torch.isnan(X_intact)) ^ (~torch.isnan(X))).type(torch.float32)
    missing_mask = (~torch.isnan(X)).type(torch.float32)
    # replace missing values with 0
    X_intact = torch.nan_to_num(X_intact, nan=nan)
    X = torch.nan_to_num(X, nan=nan)

    return dict(X_intact=X_intact,
                X=X,
                missing_mask=missing_mask,
                indicating_mask=indicating_mask)

def create_missing(train_rate, test_rate, train_X_intact, val_X_intact, test_X_intact, train_pattern, test_pattern, **kwargs):

    if train_pattern == 'point':
        train_X = mcar(train_X_intact, train_rate)
        val_X = mcar(val_X_intact, train_rate)
    elif train_pattern == 'subseq':
        train_X = seq_missing(train_X_intact, train_rate, seq_len=kwargs['sub_seq_len'])
        val_X = seq_missing(val_X_intact, train_rate, seq_len=kwargs['sub_seq_len'])
    elif train_pattern == 'block':
        train_X = block_missing(train_X_intact, factor=kwargs['block_mr'], block_len=kwargs['block_len_width'], block_width=kwargs['block_len_width'])
        val_X = block_missing(val_X_intact, factor=kwargs['block_mr'], block_len=kwargs['block_len_width'], block_width=kwargs['block_len_width'])

    if test_pattern == 'point':
        test_X = mcar(test_X_intact, test_rate)
    elif test_pattern == 'subseq':
        test_X = seq_missing(test_X_intact, test_rate, seq_len=kwargs['sub_seq_len'])
    elif test_pattern == 'block':
        test_X = block_missing(test_X_intact, factor=kwargs['block_mr'], block_len=kwargs['block_len_width'], block_width=kwargs['block_len_width'])

    return train_X, val_X, test_X

def get_mask(X: Union[np.ndarray, torch.Tensor],
             X_intact: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    X, X_intact = specify_type([X, X_intact], specify_type='tensor')

    indicating_mask = ((~torch.isnan(X_intact)) ^ (~torch.isnan(X))).type(torch.float32)
    missing_mask = (~torch.isnan(X)).type(torch.float32)

    return missing_mask, indicating_mask


def padding(nan: Union[int, float, str],
            X: Union[np.ndarray, torch.Tensor] = None,
            X_intact: Union[np.ndarray, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # nan = specify_type([nan, ], specify_type='tensor')
    term_X = specify_type([X.copy(), ], specify_type='tensor')
    term_X_intact = specify_type([X_intact.copy(), ], specify_type='tensor')
    nan_mask_X = torch.isnan(term_X)
    nan_mask_X_intact = torch.isnan(term_X_intact)

    if isinstance(nan, str):
        assert nan in ['mean', "sample_mean", 'linear', 'coeffs_linear']

        if nan == 'mean':
            mean = torch.nanmean(rearrange(term_X, 'b t c -> (b t) c'), dim=0, keepdim=True)
            if torch.isnan(mean).any():
                mean = torch.nan_to_num(mean, nan=0.0)
            mean = repeat(mean, "1 c -> b t c", t=X.shape[1], b=X.shape[0])
            term_X[nan_mask_X] = mean[nan_mask_X]
            term_X_intact[nan_mask_X_intact] = mean[nan_mask_X_intact]

        elif nan == 'sample_mean':
            sample_mean = torch.nanmean(term_X, dim=1, keepdim=True)
            if torch.isnan(sample_mean).any():
                sample_mean = torch.nan_to_num(sample_mean, nan=0.0)
            sample_mean = repeat(sample_mean, 'b 1 c -> b t c', t=X.shape[1])
            term_X[nan_mask_X] = sample_mean[nan_mask_X]
            term_X_intact[nan_mask_X_intact] = sample_mean[nan_mask_X_intact]

        elif nan == 'linear':
            pd_X = rearrange(term_X, 'b t c -> (b t) c')
            pd_X = pd.DataFrame(pd_X.cpu().numpy())
            pd_X = torch.from_numpy(pd_X.interpolate(method='linear', limit_direction='both').values)
            pd_X = rearrange(pd_X, '(b t) c -> b t c', t=X.shape[1])
            term_X[nan_mask_X] = pd_X[nan_mask_X]
            term_X_intact[nan_mask_X_intact] = pd_X[nan_mask_X_intact]
        elif nan == 'coeffs_linear':
            t_X = rearrange(term_X, 'b t c -> (b t) c')
            t_X = torchcde.linear_interpolation_coeffs(t_X)
            t_X = rearrange(t_X, '(b t) c -> b t c', t=X.shape[1])
            term_X[nan_mask_X] = t_X[nan_mask_X]
            term_X_intact[nan_mask_X_intact] = t_X[nan_mask_X_intact]
    else:
        term_X_intact = torch.nan_to_num(term_X_intact, nan=nan)
        term_X = torch.nan_to_num(term_X, nan=nan)

    term_X = specify_type([term_X, ], specify_type='tensor').clone()
    term_X_intact = specify_type([term_X_intact, ], specify_type='tensor').clone()

    return term_X_intact, term_X
