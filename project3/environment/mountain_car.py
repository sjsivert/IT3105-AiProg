

from copy import deepcopy
from math import cos, pi
from pathlib import Path
from random import uniform

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
from abstract_classes.environment import Environment
from enums import Action


class MountainCar(Environment):

    def __init__(self, goal_position: float = 0.6, position_bounds: tuple = (-1.2, 0.6), start_position_range: tuple = (-0.6, -0.4), max_velocity: float = 0.07, slope_resolution: float = 0.01, max_actions: int = 1000) -> None:
        self.goal_position = goal_position
        self.position_bounds = position_bounds
        self.start_position_range = start_position_range
        self.max_velocity = max_velocity
        self.max_actions = max_actions

        self.reset()
        self.xs, self.ys = self.__generate_slope(slope_resolution)

    def execute_action(self, action: Action) -> None:
        # Calculate velocity
        velocity = self.__compute_velocity(self.velocity, self.position, action)

        # Apply upper and lower bounds
        self.velocity = min(velocity, self.max_velocity) if velocity > 0 else max(velocity, -self.max_velocity)
        assert -self.max_velocity <= self.velocity <= self.max_velocity

        # Update position
        self.position += self.velocity

        self.position_sequence.append(self.position)

        if self.check_win_condition():
            return 1
        elif len(self.position_sequence) >= self.max_actions:
            return -1
        elif self.position <= self.position_bounds[0]:
            return -1
        else:
            return -0.002

    def check_win_condition(self) -> bool:
        return True if self.position >= self.goal_position else False

    def get_state(self) -> tuple:
        return self.position, self.velocity

    def reset(self) -> None:
        self.position = self.__get_start_position()
        self.velocity = 0

        self.position_sequence = []

    def visualize(self) -> None:
        # plt.style.use('ggplot')

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.xs, self.ys)
        #ax.add_patch(Rectangle((self.position, self.compute_y(self.position)), 0.1, 0.2, facecolor='grey'))
        ax.add_patch(plt.Circle((self.position, self.__compute_y(self.position)), 0.05, color='grey'))
        plt.show()

    def animate(self, position_sequence: list, save: bool = False) -> None:
        # plt.style.use('ggplot')

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.xs, self.ys)
        #ax.add_patch(Rectangle((self.position, self.compute_y(self.position)), 0.1, 0.2, facecolor='grey'))
        car = plt.Circle((self.position, self.__compute_y(self.position)), 0.05, color='grey')
        ax.add_patch(car)

        def __update_animation(i) -> None:
            position = position_sequence[i]
            car.center = (position, self.__compute_y(position))

        animation = FuncAnimation(fig, __update_animation, interval=10, frames=len(position_sequence), repeat=False)

        if save:
            Path("animations").mkdir(parents=True, exist_ok=True)
            animation.save('animations/output.mp4')

        plt.draw()
        plt.show()

    def __get_start_position(self) -> float:
        return uniform(self.start_position_range[0], self.start_position_range[1])

    def __generate_slope(self, resolution: float) -> None:
        xs = []
        ys = []
        for x in np.arange(self.position_bounds[0], self.position_bounds[1], resolution):
            xs.append(x)
            ys.append(self.__compute_y(x))

        return xs, ys

    @staticmethod
    def __compute_y(position: float) -> float:
        return cos(3 * (position + pi / 2))

    @staticmethod
    def __compute_velocity(velocity: float, position: float, action: Action) -> float:
        return velocity + 0.001 * action.value - 0.0025 * cos(3 * position)
