import sys

sys.path.append("./")

import pickle

from omegaconf import OmegaConf

from scripts.rf.utils import train


def main():
    print("==================  load  config  ==================")
    config = OmegaConf.load(
        open("scripts/rf/config.yaml", "r")
    )
    model = train(
        config.data,
        config.model,
        config.optim,
        config.seed
    )
    print("==================  save   model  ==================")
    with open(config.model.ckp, 'wb') as f:  
        pickle.dump(model, f)
    print("save model to " + config.model.ckp)

if __name__ == "__main__":
    main()