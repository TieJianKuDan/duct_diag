import sys

sys.path.append("./")

from core.reg.unet_pl import UnetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=UnetPL,
        config_dir="scripts/reg/config.yaml",
        log_dir="./logs/era5",
        log_name="regress",
        ckp_dir="ckps/era5/regress"
    )
    cc.test("ckps/era5/regress/epoch=82-step=259644.ckpt")

if __name__ == "__main__":
    main()
    