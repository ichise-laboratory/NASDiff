from data_processing.data_base import Dataset_Base
import torch


class DatasetFor_Transformer(Dataset_Base):
    def __init__(
        self,
        data_dict: dict,
        DEVICE: torch.device = None,
    ):
        super().__init__(data_dict=data_dict, DEVICE=DEVICE)
