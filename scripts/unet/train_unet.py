import sys

sys.path.append("./")

from core.unet.unet_pl import UnetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=UnetPL,
        config_dir="scripts/unet/config.yaml",
        log_dir="./logs/",
        log_name="unet",
        ckp_dir="ckps/unet/"
    )
    cc.train("ckps/unet/unet-1993-2022.ckpt")

if __name__ == "__main__":
    main()
    