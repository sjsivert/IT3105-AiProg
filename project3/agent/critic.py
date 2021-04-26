

import sys
import numpy as np
import torch
import torch.nn as nn

from utils.instantiate import instantiate_activation_func, instantiate_optimizer, instantiate_loss_func


class Critic(nn.Module):

    def __init__(self, epochs: int = 1, learning_rate: float = 0.001, nn_dimensions: list = [125, 256, 256, 1], activation_functions: list = ['tanh', 'tanh', 'linear'], optimizer: str = 'adam', loss_function: str = 'mse', discount_factor: float = 0.9, decay_rate: float = 0.9) -> None:
        super(Critic, self).__init__()

        assert len(nn_dimensions) - 1 == len(activation_functions)

        self.eligibilities = []
        self.state_values = {}

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.decay_rate = decay_rate

        self.__init_model(nn_dimensions, activation_functions, optimizer, loss_function)

        print(self.model)

    def __init_model(self, nn_dimensions: list, activation_functions: list, optimizer: str, loss_funciton: str) -> None:
        self.model = nn.Sequential()
        for x in range(1, len(nn_dimensions)):
            self.model.add_module(f'L{x-1}', nn.Linear(nn_dimensions[x - 1], nn_dimensions[x]))

            activation_func = instantiate_activation_func(activation_functions[x - 1])
            self.model.add_module(f'A{x-1}', activation_func) if activation_func else None

        # self.model.apply(self.init_weights)
        self.optimizer = instantiate_optimizer(optimizer, self.model.parameters(), self.learning_rate)
        self.loss_function = instantiate_loss_func(loss_funciton)

    def compute_temporal_difference_error(self, state: np.ndarray, next_state: np.ndarray, reinforcement: int) -> float:
        state_value = float(self.get_state_value(state))
        next_state_value = float(self.get_state_value(next_state))

        return reinforcement + self.discount_factor * next_state_value - state_value

    def update_state_values(self, td_error: float, reinforcement: float, state: list, next_state: list) -> None:
        state, next_state, discount_factor, reinforcement = self.__convert_to_tensors(state, next_state, self.discount_factor, reinforcement)

        target_value = torch.add(reinforcement, torch.multiply(discount_factor, self.model(next_state)))

        prediction = self.model(state)

        loss = self.loss_function(prediction, target_value)
        self.optimizer.zero_grad()
        loss.backward()

        # Modify gradients
        for index, value in enumerate(self.model.parameters()):
            value.grad *= 0.5 / td_error
            self.eligibilities[index] += value.grad
            value.grad = self.eligibilities[index] * td_error

        self.optimizer.step()

    def get_state_value(self, state: np.ndarray):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            return self.model(state)

    def reset_eligibilities(self) -> None:
        self.eligibilities.clear()
        for var in self.model.parameters():
            self.eligibilities.append(torch.zeros_like(var))

    def decay_eligibilities(self) -> None:
        for i in range(len(self.eligibilities)):
            self.eligibilities[i] = self.decay_rate * self.discount_factor * self.eligibilities[i]

    def set_eligibility(self, state: tuple, value: int) -> None:
        pass

    @staticmethod
    def __convert_to_tensors(state: list, next_state: list, discount_factor: float, reinforcement: float) -> tuple:
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        discount_factor = torch.as_tensor(discount_factor)
        reinforcement = torch.as_tensor(reinforcement)

        return state, next_state, discount_factor, reinforcement

    @ staticmethod
    def init_weights(m: nn.Module) -> None:
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
