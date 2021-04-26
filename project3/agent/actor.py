import random
import sys

from enums import Action
from utils.config_parser import Config


class Actor:

    def __init__(self, learning_rate, disc_factor, trace_decay):
        self.policy = []
        for i in range(Config.grids):
            self.policy.append({})
        self.eligibilities = {}
        self.learning_rate = learning_rate
        self.disc_factor = disc_factor
        self.trace_decay = trace_decay
        self.grid_offset = Config.offsets
        self.tile_size = Config.tile_size
        self.actions = [-1, 0, 1]

    def get_next_action(self, state, epsilon):
        if random.random() < epsilon:   # Do random action, probable if high epsilon
            return Action(random.randint(-1, 1))

        state = (state[0] + abs(Config.position_bounds[0]), state[1] + Config.max_velocity)

        action_distribution = [0, 0, 0]  # action -1, 0, +1
        #print("State: ", state)
        for grid_num in range(len(self.grid_offset)):
            #print(" --- ")
            tile_key = self.get_tile_key(state, grid_num)
            # print(tile_key)
            #print("state:   ", state)
            #print("tileKey: ", tileKey)
            policy = self.get_policy(grid_num, tile_key)
            #print(grid_num, " policy: ", policy)
            for i in range(len(action_distribution)):
                action_distribution[i] += policy[i] * 1 / len(self.grid_offset)
        #print(action_distribution, "\n----------- ")

        index = action_distribution.index(max(action_distribution))
        return Action(index - 1)

    def get_policy(self, grid_num, tile_key):
        self.init_policy(grid_num, tile_key)
        return self.policy[grid_num][tile_key]

    def get_tile_key(self, state: tuple, grid_num: int) -> tuple:
        offset = [self.grid_offset[grid_num][0] * self.tile_size[0], self.grid_offset[grid_num][1] * self.tile_size[1]]
        return ((state[0] - offset[0]) - ((state[0] - offset[0]) % self.tile_size[0]) + offset[0], (state[1] - offset[1]) - ((state[1] - offset[1]) % self.tile_size[1]) + offset[1])

    def update_policies(self, state_action_pairs: list, td_error: float) -> None:
        for state, action in state_action_pairs:
            for grid_num in range(len(self.grid_offset)):
                tile_key = self.get_tile_key(state, grid_num)
                self.init_policy(grid_num, tile_key)
                self.policy[grid_num][tile_key][action.value + 1] += self.learning_rate * td_error  # * self.eligibility[state]

    def init_policy(self, grid_num: int, tile_key: int) -> None:
        if tile_key not in self.policy[grid_num]:
            self.policy[grid_num][tile_key] = [1, 1, 1]

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()

    # Bruker ikke eligibilities i actor atm, må evt. representere eligibilities med diskret state
    def decay_eligibilities(self, state: tuple, action: Action) -> None:
        self.eligibilities[state][action] = self.eligibilities[state][action] * self.trace_decay * self.disc_factor

    def set_eligibility(self, state: tuple, action: Action) -> None:
        if state not in self.eligibilities:
            self.eligibilities[state] = [0, 0, 0]
        self.eligibilities[state][action] = 1
