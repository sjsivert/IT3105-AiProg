from enums import Action
from environment.mountain_car import MountainCar


def main():

    env = MountainCar()

    for x in range(1000):
        env.execute_action(Action.FORWARD)

    env.animate(env.position_sequence, False)


if __name__ == "__main__":
    main()
