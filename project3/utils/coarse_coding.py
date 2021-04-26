

import sys
import numpy as np


class CoarseEncoder:
    def __init__(self, grids: int, grid_offsets: list, tile_size: list, position_bounds: tuple, max_velocity: float) -> None:
        self.grids = grids
        self.tile_size = tile_size
        self.position_bounds = position_bounds
        self.max_velocity = max_velocity

        self.grid_offsets = grid_offsets

    def coarse_encode_one_hot(self, state: tuple) -> np.ndarray:
        state = (state[0] + abs(self.position_bounds[0]), state[1] + self.max_velocity)

        encoded_state = np.zeros((self.grids, 2, self.tile_size[0]))

        for grid_num in range(self.grids):
            coords: np.ndarray = self.__get_tile_index(state, grid_num, self.grid_offsets, self.tile_size)
            one_hot = self.__get_one_hot(coords, self.tile_size[0])
            encoded_state[grid_num] = one_hot

        return encoded_state.flatten()

    @staticmethod
    def __get_tile_index(state, grid_num: int, grid_offsets: list, tile_size: list) -> np.ndarray:
        offset = [grid_offsets[grid_num][0] * tile_size[0], grid_offsets[grid_num][1] * tile_size[1]]
        return np.array([(state[0] - offset[0]) - ((state[0] - offset[0]) % tile_size[0]) + offset[0], (state[1] - offset[1]) - ((state[1] - offset[1]) % tile_size[1]) + offset[1]], dtype='int')

    @staticmethod
    def __get_one_hot(values, classes) -> np.ndarray:
        res = np.eye(classes)[np.array(values).reshape(-1)]
        return res.reshape(list(values.shape) + [classes])
