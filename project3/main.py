from agent.reinforcement_learning import ReinforcementLearning
from enums import Action
from environment.mountain_car import MountainCar


def main():

    rl = ReinforcementLearning()
    rl.fit()


if __name__ == "__main__":
    main()
