

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

        critic = Critic(Config.epochs, Config.critic_learning_rate, Config.nn_dimensions, Config.activation_functions, Config.optimizer, Config.loss_function, Config.critic_discount_factor, Config.critic_decay_rate)
        actor = Actor(Config.actor_learning_rate, Config.actor_discount_factor, Config.actor_decay_rate)

        encoder = CoarseEncoder(Config.grids, Config.offsets, Config.tile_size, Config.position_bounds, Config.max_velocity)

        # Exploration vs. exploitation configuration
        initial_epsilon = Config.epsilon
        epsilon_decay = initial_epsilon / Config.episodes

        # Statistics
        training_wins = 0
        test_wins = 0
        losses = 0

        for episode in range(Config.episodes + Config.test_episodes):
            env.reset()
            critic.reset_eligibilities()
            # actor.reset_eligibilities()

            state_action_pairs = []

            if episode >= Config.episodes:
                # No exploration during final model test
                epsilon = 0
            else:
                epsilon = initial_epsilon - epsilon_decay * (episode + 1)

            state: tuple = env.get_state()
            action: Action = actor.get_next_action(state, epsilon)

            while True:
                # print(action)
                state_action_pairs.append((env.get_state(), action))
                reinforcement = env.execute_action(action)

                next_state: tuple = env.get_state()
                next_action: Action = actor.get_next_action(next_state, epsilon)

                #actor.set_eligibility(state, action)
                td_error = critic.compute_temporal_difference_error(encoder.coarse_encode_one_hot(state), encoder.coarse_encode_one_hot(next_state), reinforcement)

                critic.set_eligibility(state, 1)

                # For all (s,a) pairs
                critic.update_state_values(td_error, reinforcement, encoder.coarse_encode_one_hot(state), encoder.coarse_encode_one_hot(next_state))
                critic.decay_eligibilities()
                actor.update_policies(state_action_pairs, td_error)
                # actor.decay_eligibilities()

                if env.check_win_condition():
                    if episode < Config.episodes:
                        training_wins += 1
                    else:
                        test_wins += 1
                    break

                if env.position <= env.position_bounds[0]:
                    print('Wrong exit')
                    losses += 1
                    break

                if len(env.position_sequence) >= Config.max_actions:
                    losses += 1
                    break

                state = next_state
                action = next_action

            visualize = True
            if visualize:

                if episode < Config.episodes:
                    print(f'Episode: {episode}, wins: {training_wins}, losses: {losses}, epsilon: {round(epsilon, 5)}')
                if episode == Config.episodes:
                    print(f'Testing final model...')

        print(f'Final model win rate: {test_wins}/{Config.test_episodes} = {round(test_wins/Config.test_episodes*100, 2)}% ')
