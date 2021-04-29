import random
import sys
from agent.reinforcement_learning import ReinforcementLearning
from enums import Action
from environment.mountain_car import MountainCar
from utils.coarse_coding import CoarseEncoder
from utils.config_parser import Config


def main():
    """ encoder = CoarseEncoder(Config.grids, Config.offsets, Config.tile_size, Config.position_bounds, Config.max_velocity)

    one_hot = encoder.coarse_encode_one_hot((0.3, 0.015))

    encoder.visualize_grids() """

    """ env = MountainCar(Config.goal_position, Config.position_bounds, Config.start_position_range, Config.max_velocity, Config.slope_resolution, Config.max_actions)

    for x in range(1000):
        env.execute_action(Action(random.randint(-1, 1)))

    env.animate(env.position_sequence)
    sys.exit() """

    rl = ReinforcementLearning()
    rl.fit()


if __name__ == "__main__":
    main()
