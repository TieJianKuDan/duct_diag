import gc

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from .ds import *


class PLDM(pl.LightningDataModule):
    '''
    abstract class
    '''
    def __init__(self):
        super(PLDM, self).__init__()
    
    @property
    def train_sample_num(self):
        return len(self.train_set)
    
    @property
    def val_sample_num(self):
        return len(self.val_set)
    
    @property
    def test_sample_num(self):
        return len(self.test_set)

    def train_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.train_set, 
                batch_size=self.batch_size, 
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
        return dl

    def val_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.val_set, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.val_set, 
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers
            )
        return dl

    def test_dataloader(self):
        dl = None
        if self.num_workers == 1:
            dl = DataLoader(
                self.test_set,
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True
            )
        else:
            dl = DataLoader(
                self.test_set,
                batch_size=self.batch_size, 
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers
            )
        return dl

class ERA5PLDM(PLDM):
    """
    ouput shape: ((b, 6, h, w), (b, 1, h, w))
    """
    def __init__(self, data_config, batch_size, seed):
        super(ERA5PLDM, self).__init__()
        self.edh_dir = data_config.edh_dir
        self.era5_dir = data_config.era5_dir
        self.val_ratio = data_config.val_ratio
        self.num_workers = data_config.num_workers
        self.persistent_workers = data_config.persistent_workers
        self.batch_size = batch_size
        self.seed = seed
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage:str):
        datasets = None
        if stage == "fit" \
            and (
                self.train_set == None \
                or self.val_set == None
            ):
            print("prepare train_set and val_set")
            datasets = ERA5Dataset(
                self.edh_dir, self.era5_dir, flag=stage
            )
            self.train_set, self.val_set = random_split(
                datasets, 
                [
                    1 - self.val_ratio, 
                    self.val_ratio,
                ], 
                generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test" \
            and self.test_set == None:
            print("prepare test_set")
            datasets = ERA5Dataset(
                self.edh_dir, self.era5_dir, flag=stage
            )
            self.test_set = datasets
        else:
            print(stage)
            return
        self.lon = datasets.lon
        self.lat = datasets.lat
        self.time_seq = datasets.time
        gc.collect()
 