import sys

from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearnex import patch_sklearn, unpatch_sklearn

sys.path.append("./")

from time import time

import numpy as np
from einops import rearrange
from sklearn.model_selection import train_test_split
from sklearn import metrics

from scripts.utils.ds import ERA5Dataset

def prepare_data(edh_path, era5_path, seed, flag="train"):
    print("loading...")
    ts = time()
    ds_train = ERA5Dataset(
        edh=edh_path,
        era5=era5_path,
        flag=flag
    )
    te = time()
    print(f"complete loading and spend {te - ts}s")
    data = np.array([
        ds_train.u10, 
        ds_train.v10, 
        ds_train.t2m,
        ds_train.msl, 
        ds_train.sst, 
        ds_train.q2m
    ])
    data = rearrange(data, "c b h w -> (b h w) c")
    labels = np.expand_dims(ds_train.edh, axis=0)
    labels = rearrange(labels, "c b h w -> (b h w) c")
    train_test_set = train_test_split(
        data, labels, test_size=0.1, random_state=seed
    )

    return train_test_set

def train(data_cfg, model_cfg, optim_cfg, seed):
    patch_sklearn()
    print("==================  prepare data  ==================")
    data_t, data_v, label_t, label_v = prepare_data(
        data_cfg.edh, data_cfg.era5, seed, "train"
    )
    params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True
    )
    print("================== start to train ==================")
    ts = time()
    rf = RandomForestRegressor(
        optim_cfg.n_estimators,
        **params
    )
    rf.fit(data_t, label_t.reshape(-1)) 
    te = time()
    print(f"complete training and spend {te - ts}s")
    pred = rf.predict(data_v)
    print(
        f"mse: {metrics.mean_squared_error(pred, label_v)}"
    )
    unpatch_sklearn()
    return rf