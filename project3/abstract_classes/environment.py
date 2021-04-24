from abc import ABC, abstractmethod
from argparse import Action


class Environment(ABC):
    """
    Abstract class implemented by each environment
    """

    @abstractmethod
    def execute_action(self, action: Action) -> None:
        pass

    @abstractmethod
    def check_win_condition(self) -> bool:
        pass

    @abstractmethod
    def get_state(self) -> tuple:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def visualize(self) -> None:
        pass

    @abstractmethod
    def animate(self, position_sequence: list, save: bool = False) -> None:
        pass
