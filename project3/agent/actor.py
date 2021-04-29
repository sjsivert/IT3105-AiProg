import random
import sys

import numpy as np

from enums import Action
from utils.config_parser import Config


class Actor:

    def __init__(self, learning_rate, discount_factor, decay_rate):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate

        self.policies = {}
        self.eligibilities = {}

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()

    def set_eligibility(self, state: np.ndarray, action: Action, value: int) -> None:
        self.eligibilities.setdefault(str(state), {})[str(action.value)] = value

    def generate_action(self, state: np.ndarray, epsilon: float) -> Action:
        if random.uniform(0, 1) < epsilon:
            return Action(random.randint(-1, 1))

        else:
            action_policies = {}
            for action in Action:
                action = action.value

                self.policies.setdefault(str(state), {}).setdefault(str(action), 0)
                action_policies[action] = self.policies[str(state)][str(action)]

            max_value = max(action_policies.values())
            keys = [k for k, v in action_policies.items() if v == max_value]

            chosen_action = random.choice(keys)

        return Action(int(chosen_action))

    def update_policies(self, td_error: float) -> None:
        for state_key, action_dict in self.eligibilities.items():
            for action_key, eligibility in action_dict.items():
                value = self.policies.setdefault(str(state_key), {}).setdefault(str(action_key), 0)

                self.policies[str(state_key)][str(action_key)] = value + self.learning_rate * eligibility * td_error

    def decay_eligibilities(self) -> None:
        for state_key, action_dict in self.eligibilities.items():
            for action_key, eligibility in action_dict.items():
                self.eligibilities[str(state_key)][str(action_key)] = self.decay_rate * self.discount_factor * eligibility
