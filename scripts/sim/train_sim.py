import sys

sys.path.append("./")

from core.unet.unet_pl import SimUNetPL
from scripts.utils.classic import ClassicConf
from scripts.utils.dm import ERA5PLDM


def main():
    cc = ClassicConf(
        dm_class=ERA5PLDM,
        model_class=SimUNetPL,
        config_dir="scripts/sim/config.yaml",
        log_dir="./logs/",
        log_name="sim",
        ckp_dir="ckps/sim/"
    )
    cc.train()

if __name__ == "__main__":
    main()
    