
from copy import deepcopy
from turtle import position
from matplotlib import pyplot as plt
import numpy as np
from agent.actor import Actor
from agent.critic import Critic
from enums import Action
from environment.mountain_car import MountainCar
from utils.coarse_coding import CoarseEncoder
from utils.config_parser import Config


class ReinforcementLearning:

    def __init__(self) -> None:
        pass

    def fit(self):
        env = MountainCar(Config.goal_position, Config.position_bounds, Config.start_position_range, Config.max_velocity, Config.slope_resolution, Config.max_actions)

        critic = Critic(Config.critic_learning_rate, Config.nn_dimensions, Config.activation_functions, Config.optimizer, Config.loss_function, Config.critic_discount_factor, Config.critic_decay_rate)
        actor = Actor(Config.actor_learning_rate, Config.actor_discount_factor, Config.actor_decay_rate)

        encoder = CoarseEncoder(Config.grids, Config.offsets, Config.tile_size, Config.position_bounds, Config.max_velocity)

        # Exploration vs. exploitation configuration
        initial_epsilon = Config.epsilon
        epsilon_decay = initial_epsilon / Config.episodes

        # Statistics
        training_wins = 0
        test_wins = 0
        losses = 0
        final_positions = []

        position_sequences = {}

        for episode in range(Config.episodes + Config.test_episodes):
            env.reset()
            critic.reset_eligibilities()
            actor.reset_eligibilities()

            state_action_pairs = []

            if episode >= Config.episodes:
                # No exploration during final model test
                epsilon = 0
            else:
                epsilon = max(0, initial_epsilon - epsilon_decay * (episode + 1))

            state: tuple = env.get_state()
            state_coarse: np.ndarray = encoder.coarse_encode_one_hot(state)
            action: Action = actor.generate_action(state_coarse, epsilon)

            while True:
                state_action_pairs.append((env.get_state(), action))
                reinforcement = env.execute_action(action)

                next_state: tuple = env.get_state()
                next_state_coarse: np.ndarray = encoder.coarse_encode_one_hot(state)
                next_action: Action = actor.generate_action(next_state_coarse, epsilon)

                actor.set_eligibility(state_coarse, action, 1)
                td_error = critic.compute_temporal_difference_error(state_coarse, next_state_coarse, reinforcement)

                critic.set_eligibility(state_coarse, 1)

                # For all (s,a) pairs
                critic.update_state_values(td_error, reinforcement, state_coarse, next_state_coarse)
                critic.decay_eligibilities()
                actor.update_policies(td_error)
                actor.decay_eligibilities()

                if env.check_win_condition():
                    if episode < Config.episodes:
                        training_wins += 1
                    else:
                        test_wins += 1
                        position_sequences[episode - Config.episodes] = deepcopy(env.position_sequence)
                    break

                if len(env.position_sequence) >= Config.max_actions:
                    losses += 1
                    break

                state = next_state
                state_coarse = next_state_coarse
                action = next_action

            final_positions.append(env.best_position)

            if episode < Config.episodes:
                print(f'Episode: {episode}, wins: {training_wins}, losses: {losses}, epsilon: {round(epsilon, 5)}, best_pos: {env.best_position:.5}, best_vel: {env.best_velocity:.5}')
            if episode == Config.episodes:
                print(f'Testing final model...')

        print(f'Final model win rate: {test_wins}/{Config.test_episodes} = {round(test_wins/Config.test_episodes*100, 2)}% ')
        self.visualize_progress(final_positions, Config.position_bounds)

        for key, seq in position_sequences.items():
            env.animate(seq, True, key)

    def visualize_progress(self, final_positions: list, position_bounds: tuple) -> None:
        plt.figure(figsize=(24, 12))
        plt.xlabel('episodes')
        plt.ylabel('Position')
        plt.ylim(ymin=position_bounds[0], ymax=position_bounds[1])
        plt.plot(final_positions)
        plt.show()
