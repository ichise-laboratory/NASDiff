from data_processing.data_base import Dataset_Base
import torch
from typing import Union

class DatasetFor_CSDI(Dataset_Base):
    def __init__(self,
                 data_dict: dict,
                 DEVICE: torch.device = None):
        super().__init__(data_dict=data_dict,
                         DEVICE=DEVICE)

        observed_tp = torch.arange(0, self.seq_len, dtype=torch.float32)
        self.data_dict['observed_tp'] = observed_tp
        self.to_device(DEVICE)

    def getitem_core(self, item):
        temp: dict[str, Union[torch.Tensor, int]] = dict(
            sample_idx=item,
            X_intact=self.data_dict['X_intact'][item].transpose(-1, -2),
            X=self.data_dict['X'][item].transpose(-1, -2),
            missing_mask=self.data_dict['missing_mask'][item].transpose(-1, -2),
            indicating_mask=self.data_dict['indicating_mask'][item].transpose(-1, -2),
            observed_tp=self.data_dict['observed_tp'])
        return temp
