import sys

import numpy as np
from einops import rearrange

sys.path.append("./")

import torch
from time import time
import xgboost as xgb
from omegaconf import OmegaConf
from sklearn import metrics
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from core.reg.unet_pl import UnetPL
from scripts.utils.dm import ERA5Dataset


def test_UNet():
    '''
    evaluate the perfomance of UNet in regression
    '''
    config = OmegaConf.load(open("scripts/reg/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = ERA5Dataset(R"data/test/edh", R"data/test/era5")
    test_loader = DataLoader(test_set, 128, True)

    # model
    net = UnetPL.load_from_checkpoint(
        R"ckps/era5/regress/epoch=82-step=259644.ckpt"
    ).eval()
    loss_total = 0
    for i, (era5, edh) in enumerate(test_loader):
        era5 = era5.to(net.device)
        edh = edh.to(net.device)
        edh_hat = net(era5)
        loss = mse_loss(edh_hat, edh, reduction="mean")
        print(loss.item())
        loss_total += loss.item()
        print(i)
    print(loss_total / (i+1))
    # fig = edh_subplot(
    #     test_set.lon,
    #     test_set.lat,
    #     torch.cat(
    #         (
    #             edh.detach().cpu().squeeze(1), 
    #             edh_hat.detach().cpu().squeeze(1)
    #         ),
    #         0
    #     ),
    #     2,
    #     8
    # )
    # fig.savefig("imgs/regress.jpg")

def test_XGBoost():
    '''
    evaluate XGboost in regression
    '''
    params = {"device": "cuda"}
    model = xgb.Booster(params)
    model.load_model("ckps/era5/regress/xgboost_1993.model")
    # Data
    year = 1993
    ds_train = ERA5Dataset(
        edh=f"data/train/edh/{year}",
        era5=f"data/train/era5/{year}"
    )
    features = np.array([
        ds_train.u10, 
        ds_train.v10, 
        ds_train.t2m,
        ds_train.msl, 
        ds_train.sst, 
        ds_train.q2m
    ])
    features = rearrange(features, "c b h w -> (b h w) c")
    labels = np.expand_dims(ds_train.edh, axis=0)
    labels = rearrange(labels, "c b h w -> (b h w) c")
    fl_test = xgb.DMatrix(features)
    ts = time()
    pred = model.predict(fl_test)
    te = time()
    print(f"{(te - ts):.3f}s")
    mse = metrics.mean_squared_error(pred, labels)
    print(mse)

if __name__ == "__main__":
    test_XGBoost()