import sys

sys.path.append("./")

from time import time

import numpy as np
import xgboost as xgb
from einops import rearrange
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from scripts.utils.ds import ERA5Dataset, ERA5Iterator


def prepare_data(edh_path, era5_path, seed, batch_size, flag="train"):
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
    data_t, data_v, labels_t, labels_v = train_test_set

    dtrain = ERA5Iterator(
        data_t,
        labels_t,
        batch_size=batch_size
    )
    dval = ERA5Iterator(
        data_v,
        labels_v,
        batch_size=batch_size
    )
    return xgb.DMatrix(dtrain), xgb.DMatrix(dval)

def train(data_cfg, model_cfg, optim_cfg, seed):
    print("==================  prepare data  ==================")
    dtrain, dval = prepare_data(
        data_cfg.edh, data_cfg.era5,
        seed, data_cfg.batch_size, "train"
    )
    params = OmegaConf.to_container(
        model_cfg.params,
        resolve=True
    )
    print("================== start to train ==================")
    ts = time()
    model = xgb.train(
        params, dtrain,
        num_boost_round=optim_cfg.n_estimators,
        early_stopping_rounds=optim_cfg.early_stopping,
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=optim_cfg.verbose_eval
    )
    te = time()
    print(f"complete training and spend {te - ts}s")
    return model

def search_params():
    # Data
    ds_train = ERA5Dataset(
        edh="data/train/edh",
        era5="data/train/era5"
    )
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
    f_train, f_val, l_train, l_val = train_test_split(
        data, labels, test_size=0.2, random_state=2333)
    fl_train = xgb.DMatrix(f_train, l_train)
    fl_val = xgb.DMatrix(f_val)

    # parameter search
    param_dict = {
        "objective": ["reg:squarederror"],
        "tree_method": ["hist"],
        "device": ["cuda"],
        "max_depth": np.arange(7, 8, 1), 
        "learning_rate": [0.1],
        "n_estimators": np.arange(5000, 11000, 1000),
        "booster": ["gbtree"],
        # "subsample": np.arange(1, 1.1, 0.1), # ↓
        # "colsample_bytree": np.arange(1, 1.1, 0.1), # ↓
        # "colsample_bylevel": np.arange(1, 1.1, 0.1), # ↓
        # "min_child_weight": np.arange(1, 1.1, 0.1), # ↑
        # "lambda": [1], # ↑
        # "alpha": [0] # ↑
    }
    param_grid = ParameterGrid(param_dict)
    best_param = None
    best_score = np.Inf
    for param in param_grid:
        print(param)
        param_ = param
        boost_round = param.pop("n_estimators")
        model = xgb.train(
            param, fl_train, 
            num_boost_round=boost_round
        )
        pred = model.predict(fl_val)
        rmse = metrics.mean_squared_error(pred, l_val, squared=False)
        print(f"RMSE: {rmse}")
        if rmse < best_score:
            best_param = param_
            best_score = rmse
    print("======= Best Param =======")
    print(best_param)