from trainer import Trainer
from parameters import Parameters as p

if __name__ == '__main__':
    trainer = Trainer(p["useDiamondBoard"], p["showFinalSolution"])
    trainer.train()
