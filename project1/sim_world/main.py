import json
from SimWorld import SimWorld
from GenerateBoard import Boardtype, BoardState, HexBoard
from Actor import Actor
from Critic import Critic
from Utils import GetRandomizedBoard, WriteTables, ReadTables, TestModel


def main():
    # Load parameters from file
    with open('project1/sim_world/parameters.json') as f:
        parameters = json.load(f)

    board_type = parameters['board_type'].lower()
    learning_rate_actor = parameters['learning_rate_actor']
    learning_rate_critic = parameters['learning_rate_critic']
    discount_factor_actor = parameters['discount_factor_actor']
    discount_factor_critic = parameters['discount_factor_critic']
    decay_rate_actor = parameters['decay_rate_actor']
    decay_rate_critic = parameters['decay_rate_critic']
    epsilon = parameters['epsilon']
    epsilon_decay_rate = parameters['epsilon_decay_rate']
    num_episodes = parameters['num_episodes']
    board_size = parameters['board_size']
    use_neural_critic = parameters['use_neural_critic']
    hidden_layers_dim = parameters['hidden_layers_dim']
    open_cells = parameters['open_cells']

    doEpisodes(
        episodes=num_episodes,
        boardSize=board_size,
        eligibilityDecayActor=decay_rate_actor,
        eligibilityDecayCritic=decay_rate_critic,
        maxRemovePegs=4,  # Why do we need this?,
        boardType=board_type,
        removePegs=open_cells,
        epsilon=epsilon,
        learningRateActor=learning_rate_actor,
        learningRateCritic=learning_rate_critic,
        discountFactorActor=discount_factor_actor,
        discountFactorCritic=discount_factor_critic,
    )

    TestModel(4, 'diamond', open_cells)


def doEpisodes(episodes, boardSize, maxRemovePegs, boardType, removePegs, eligibilityDecayActor, eligibilityDecayCritic, epsilon=0.6,
               learningRateActor=0.1, learningRateCritic=0.1, discountFactorActor=0.9, discountFactorCritic=0.9, policyTable={}, valueTable={}):
    TotalError = 0
    stepsTaken = 1

    actor = Actor(eligibilityDecayActor, learningRateActor, epsilon, policyTable,
                  discountFactorActor)
    critic = Critic(eligibilityDecayCritic, learningRateCritic, valueTable,
                    discountFactor=discountFactorCritic)

    for i in range(episodes):
        #world = GetRandomizedBoard(boardSize, maxRemovePegs, boardType)
        world = SimWorld(
            boardType=boardType,
            boardWith=boardSize,
            removeLocations=removePegs
        )

        actor.resetEligibility()
        critic.resetEligibility()
        critic.tdError = 0
        reward = 0
        state = world.stateToHash()

        chosenAction = actor.ChooseActionByPolicy(world)

        while True:
            reward = world.makeAction(chosenAction)
            nextAction = actor.ChooseActionByPolicy(world)
            nextState = world.stateToHash()

            actor.eligibility[state + str(chosenAction)] = 1
            critic.updateTDError(reward, state, nextState)
            critic.eligibility[state] = 1
            TotalError += abs(critic.tdError)
            for SAP in world.getGameLog():

                critic.updateValue(SAP)
                critic.decayEligibility(SAP)

                actor.updatePolicy(SAP, critic.tdError)
                actor.decayEligibility(SAP)

            if chosenAction == None:
                break
            chosenAction = nextAction
            state = nextState
            stepsTaken += 1

        print('Episode:', i, 'MeanError', TotalError / stepsTaken)

    WriteTables(critic.getValueTable(), actor.getPolicyTable())


if __name__ == "__main__":
    main()
