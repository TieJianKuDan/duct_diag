import sys

sys.path.append("./")
import pickle
from time import time

import numpy as np
import xgboost as xgb
from einops import rearrange
from sklearn import metrics

from scripts.utils.ds import ERA5Dataset


def main():
    # data
    print("loading...")
    ts = time()
    ds_train = ERA5Dataset(
        edh="data/test/edh",
        era5="data/test/era5",
        flag="test"
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
    # model
    with open("ckps/rf/rf-2003-2022.pkl", "rb") as f:
        model = pickle.load(f)
    # predict
    ts = time()
    pred = model.predict(data)
    te = time()
    print(f"MSE: {metrics.mean_squared_error(pred, labels)}")
    print(f"RMSE: {metrics.mean_squared_error(pred, labels, squared=False)}")
    print(f"MAE: {metrics.mean_absolute_error(pred, labels)}")
    print(f"MedAE: {metrics.median_absolute_error(pred, labels)}")
    print(f"RS: {metrics.r2_score(pred, labels)}")
    print(f"t: {te - ts}s")

if __name__ == "__main__":
    main()