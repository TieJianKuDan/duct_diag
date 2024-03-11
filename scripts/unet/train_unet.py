import sys

sys.path.append("./")

from core.reg.unet_pl import UnetPL
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
    cc.train()

if __name__ == "__main__":
    main()
    