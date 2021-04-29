
import ast
import configparser


class Config:
    config = configparser.ConfigParser()
    config.read('config.ini')

    train = bool(ast.literal_eval(config.get('PROGRAM_FLOW', 'train')))

    goal_position = float(config.get('ENVIRONMENT', 'goal_position'))
    position_bounds = tuple(ast.literal_eval(config.get('ENVIRONMENT', 'position_bounds')))
    start_position_range = tuple(ast.literal_eval(config.get('ENVIRONMENT', 'start_position_range')))
    max_velocity = float(config.get('ENVIRONMENT', 'max_velocity'))
    max_actions = int(config.get('ENVIRONMENT', 'max_actions'))
    slope_resolution = float(config.get('ENVIRONMENT', 'slope_resolution'))

    episodes = int(config.get('LEARNING', 'episodes'))
    test_episodes = int(config.get('LEARNING', 'test_episodes'))

    critic_learning_rate = float(config.get('LEARNING', 'critic_learning_rate'))
    nn_dimensions = list(ast.literal_eval(config.get('LEARNING', 'nn_dimensions')))
    activation_functions = list(ast.literal_eval(config.get('LEARNING', 'activation_functions')))
    optimizer = str(ast.literal_eval(config.get('LEARNING', 'optimizer')))
    loss_function = str(ast.literal_eval(config.get('LEARNING', 'loss_function')))
    critic_discount_factor = float(config.get('LEARNING', 'critic_discount_factor'))
    critic_decay_rate = float(config.get('LEARNING', 'critic_decay_rate'))

    actor_learning_rate = float(config.get('LEARNING', 'actor_learning_rate'))
    actor_discount_factor = float(config.get('LEARNING', 'actor_discount_factor'))
    actor_decay_rate = float(config.get('LEARNING', 'actor_decay_rate'))

    grids = int(config.get('COARSE_CODING', 'grids'))
    offsets = list(ast.literal_eval(config.get('COARSE_CODING', 'offsets')))
    tile_size = list(ast.literal_eval(config.get('COARSE_CODING', 'tile_size')))

    epsilon = float(config.get('EPSILON', 'epsilon'))
