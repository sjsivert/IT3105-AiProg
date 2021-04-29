

import sys
from matplotlib import pyplot as plt
import numpy as np


class CoarseEncoder:
    def __init__(self, grids: int, grid_offsets: list, tile_size: list, position_bounds: tuple, max_velocity: float) -> None:
        self.grids = grids
        self.tile_size = tile_size
        self.position_bounds = position_bounds
        self.max_velocity = max_velocity
        self.grid_offsets = grid_offsets

        self.pos_range = abs(self.position_bounds[0]) + abs(self.position_bounds[1])
        self.pos_interval = self.pos_range / self.tile_size[0]
        self.pos_offset = self.pos_interval / self.grids

        self.vel_range = 2 * self.max_velocity
        self.vel_interval = self.vel_range / self.tile_size[0]
        self.vel_offset = self.vel_interval / self.grids

    def coarse_encode_one_hot(self, state: tuple) -> np.ndarray:
        state = (state[0] + abs(self.position_bounds[0]), state[1] + self.max_velocity)
        encoded_state = np.zeros((self.grids, self.tile_size[0] * self.tile_size[1]))
        for grid_num in range(self.grids):

            coords: np.ndarray = self.__get_tile_coordinates(state, grid_num)
            # print(coords)
            index = coords[0] + coords[1] * self.tile_size[0]
            # print(index)
            one_hot = np.eye(self.tile_size[0] * self.tile_size[1])[index]
            # print(one_hot)
            encoded_state[grid_num] = one_hot
        return encoded_state.flatten()

    def __get_tile_coordinates(self, state, grid_num: int) -> np.ndarray:
        max_index = self.tile_size[0] - 1

        x_offset = self.pos_offset * self.grid_offsets[grid_num][0]
        pos_index = int((state[0] - x_offset) // self.pos_interval)
        pos_index = max(0, pos_index)
        pos_index = min(max_index, pos_index)

        y_offset = self.vel_offset * self.grid_offsets[grid_num][1]
        vel_index = int((state[1] - y_offset) // self.vel_interval)
        vel_index = max(0, vel_index)
        vel_index = min(max_index, vel_index)
        #print(f'x: {round(state[0], 2)} + {round(x_offset,2)} // {round(self.pos_interval,2)}')
        #print(f'y: {round(state[1], 2)} + {round(y_offset,2)} // {round(self.vel_interval,2)}')
        #print(pos_index, vel_index)

        return pos_index, vel_index

    def visualize_grids(self) -> None:
        plt.figure(figsize=(10, 10))
        colors = ['red', 'blue', 'green', 'purple', 'yellow']
        for grid_num in range(self.grids):
            for tile_num in range(self.tile_size[0] + 1):

                x_offset = self.pos_offset * self.grid_offsets[grid_num][0]
                x = self.pos_interval * tile_num + x_offset

                y_offset = self.vel_offset * self.grid_offsets[grid_num][1]
                y = self.vel_interval * tile_num + y_offset

                #print(x, y)
                plt.vlines(x=x, ymin=y_offset, ymax=self.vel_range + y_offset, color=colors[grid_num], label=grid_num)
                plt.hlines(y=y, xmin=x_offset, xmax=self.pos_range + x_offset, color=colors[grid_num], label=grid_num)

        y_padding = self.vel_range / 10
        plt.ylim(ymin=-y_padding, ymax=self.vel_range + y_padding)

        x_padding = self.pos_range / 10
        plt.xlim(xmin=-x_padding, xmax=self.pos_range + x_padding)

        # plt.legend()
        plt.show()
