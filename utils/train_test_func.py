from utils.log import My_logger
from data_processing.data_base import Data_Base

import torch
import importlib
import os
import sys

def get_model_dataset_class(models_root, model_name):
    if os.path.isabs(models_root):
        models_root = os.path.normpath(models_root)
        parent_dir = os.path.dirname(models_root)
        package_name = os.path.basename(models_root)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        module_path = f"{package_name}.{model_name}.architecture"
        dataset_path = f"{package_name}.{model_name}.data_processing"
    else:
        module_path = os.path.join(models_root, model_name, 'architecture')
        module_path = module_path.replace('/', '.')
        dataset_path = os.path.join(models_root, model_name, 'data_processing')
        dataset_path = dataset_path.replace('/', '.')
    module = importlib.import_module(module_path)

    dataset = importlib.import_module(dataset_path)
    return getattr(module, model_name), getattr(dataset, 'DatasetFor_' + model_name)

def get_dataloader(model_config,
                   dataset,
                   dataset_config,
                   exp_config,
                   Dataset_Class,
                   DEVICE
                   ):
    if model_config.get('General', False):
        if model_config['General'].get('setting', False):
            for key, value in model_config['General']['setting'].items():
                model_config[dataset]['setting'][key] = value

    model_config = model_config[dataset]['setting']

    dataset_info = dataset_config[dataset]
    dataset_info["BATCHSIZE"] = model_config['BATCHSIZE']
    dataset_info["BATCHSIZE_Test"] = exp_config['test']['BATCHSIZE_Test']
    dataset_info["selected_dataset"] = dataset
    dataset_info["padding_mode"] = exp_config['global']['padding_mode']
    dataset_info["scalar"] = exp_config['global']['scalar']
    dataset_info["DEVICE"] = DEVICE
    dataset_info["Dataset_Class"] = Dataset_Class
    dataset_info["shuffle"] = exp_config["train"].get("shuffle", True)

    if exp_config['train']['pattern'] in ['block', 'subseq']:
        dataset_info['train_missing_ratio'] = 0.5
        dataset_info['test_missing_ratio'] = 0.5
    else:
        dataset_info['train_missing_ratio'] = exp_config['train']['missing_ratio']
        dataset_info['test_missing_ratio'] = exp_config['test']['missing_ratio']
    dataset_info['train_pattern'] = exp_config['train']['pattern']
    dataset_info['test_pattern'] = exp_config['test']['pattern']

    db = Data_Base(**dataset_info)

    return  db


def get_net_object(model_config,
                   dataset,
                   net_class,
                   seq_len,
                   d_feature,
                   DEVICE):

    # update general hyperparameters to all dataset config
    if model_config.get('General', False):
        if model_config['General'].get('init', False):
            for key, value in model_config['General']['init'].items():
                model_config[dataset]['init'][key] = value

    init_param = model_config[dataset]['init']

    net = net_class(**init_param).to(DEVICE)

    return net

def get_optim(model_config,
              dataset,
              net):
    optimizer = None
    model_config = model_config[dataset]['setting']
    if model_config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=model_config['lr'])
    # if model_config['info']['model_name'] == 'OURS_LowF.yaml':
    #     optimizer = torch.optim.Adam([
    #                                     {'params': net.TF_module_1.parameters(), 'lr': model_config['lr']},
    #                                     {'params': net.lowrank_recon.parameters(), 'lr': model_config['lr']},
    #                                     # {'params': net.high_frequency_ex.parameters(), 'lr': model_config['lr']},
    #                                     # {'params': net.high_frequency_ex.high_f_embedding.parameters(), 'lr': 1e-3},
    #                                     {'params': net.high_frequency_ex.high_f_embedding.parameters(), 'lr': model_config['lr']},
    #                                     {'params': net.high_frequency_ex.low_f_embedding.parameters(), 'lr': model_config['lr']},
    #                                     # {'params': net.high_frequency_ex.layernorm.parameters(), 'lr': model_config['lr']},
    #                                     {'params': net.high_frequency_ex.flimit_h, 'lr': 1e-1},
    #                                     {'params': net.high_frequency_ex.flimit_l, 'lr': 1e-1}
    #                                 ])


    return optimizer

def get_log_info(model_config,
                 exp_config,
                 dataset,
                 seed,
                    ):
    if model_config.get(dataset, False):
        model_config = model_config[dataset]
        info = {**model_config['init'],
                'BATCHSIZE': model_config['setting']['BATCHSIZE'],
                'seed': seed,
                'dataset': dataset,
                'padding_mode': exp_config['global']['padding_mode'],
                'save_model': exp_config['global']['save_model'],
                'train_missing_ratio': exp_config['train']['missing_ratio'],
                'test_missing_ratio': exp_config['test']['missing_ratio'],
                'train_pattern': exp_config['train']['pattern'],
                'test_pattern': exp_config['test']['pattern'],
                    }
    else:
        info = {**model_config,
                'dataset': dataset,
                'padding_mode': exp_config['global']['padding_mode'],
                'save_model': exp_config['global']['save_model'],
                'train_missing_ratio': exp_config['train']['missing_ratio'],
                'test_missing_ratio': exp_config['test']['missing_ratio'],
                'train_pattern': exp_config['train']['pattern'],
                'test_pattern': exp_config['test']['pattern'],
                }
    content = My_logger.format_message_multicolumn(info, content='param')

    return content
