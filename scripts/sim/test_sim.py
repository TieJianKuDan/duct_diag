import sys


sys.path.append("./")

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from core.unet.unet_pl import SimUNetPL
from scripts.utils.dm import ERA5Dataset
from scripts.utils import metrics


def main():
    '''
    evaluate the perfomance of SimUNet in regression
    '''
    config = OmegaConf.load(open("scripts/sim/config.yaml", "r"))
    torch.set_float32_matmul_precision(
        config.optim.float32_matmul_precision
    )

    # data
    test_set = ERA5Dataset(R"data/test/edh", R"data/test/era5")
    test_loader = DataLoader(test_set, 256, True)

    # model
    net = SimUNetPL.load_from_checkpoint(
        R"ckps/era5/regress/v4.ckpt"
    ).eval()

    truth = []
    pred = []
    for _, (era5, edh) in enumerate(test_loader):
        era5 = era5.to(net.device)
        edh = edh.to(net.device)
        edh_hat = net(era5)
        truth.append(edh.cpu().detach())
        pred.append(edh_hat.cpu().detach())

    pred = torch.cat(pred, 0)
    truth = torch.cat(truth, 0)
    rmse = metrics.RMSE(pred, truth).item()
    mae = metrics.MAE(pred, truth).item()
    medae = metrics.MedAE(pred, truth).item()
    r2 = metrics.RS(pred, truth).item()
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"MedAE: {medae}")
    print(f"RS: {r2}")

if __name__ == "__main__":
    main()