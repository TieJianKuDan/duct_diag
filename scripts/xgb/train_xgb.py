import sys

sys.path.append("./")

from omegaconf import OmegaConf

from scripts.xgb.utils import train


def main():
    print("==================  load  config  ==================")
    config = OmegaConf.load(
        open("scripts/xgb/config.yaml", "r")
    )
    model = train(
        config.data,
        config.model,
        config.optim,
        config.seed
    )
    print("==================  save   model  ==================")
    model.save_model(config.model.ckp)
    print("save model to " + config.model.ckp)

if __name__ == "__main__":
    main()