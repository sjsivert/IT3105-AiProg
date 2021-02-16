import json
from sim_world.SimWorld import SimWorld
from sim_world.GenerateBoard import Boardtype, BoardState, HexBoard
from actor_critic.Actor import Actor
from actor_critic.Critic import Critic
from actor_critic.CriticNeural import CriticNeural
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
    hidden_layers_dim = parameters['hidden_layers_dim']
    critic_type = parameters['critic_type']
    open_cells = parameters['open_cells']

    doEpisodes(
        episodes=num_episodes,
        boardSize=board_size,
        eligibilityDecayActor=decay_rate_actor,
        eligibilityDecayCritic=decay_rate_critic,
        maxRemovePegs=4,  # Used for the general method when not defining the startboard
        criticType=critic_type,
        boardType=board_type,
        removePegs=open_cells,
        epsilon=epsilon,
        epsilonDecayRate=epsilon_decay_rate,
        learningRateActor=learning_rate_actor,
        learningRateCritic=learning_rate_critic,
        discountFactorActor=discount_factor_actor,
        discountFactorCritic=discount_factor_critic,
    )

    TestModel(board_size, board_type, open_cells)


def doEpisodes(episodes, boardSize, maxRemovePegs, boardType, removePegs, eligibilityDecayActor, eligibilityDecayCritic,
               epsilon=0.6, hiddenLayersDim=[50],
               learningRateActor=0.1, learningRateCritic=0.1, criticType="table", discountFactorActor=0.9, discountFactorCritic=0.9, policyTable={}, valueTable={}, epsilonDecayRate=1):
    TotalError = 0
    stepsTaken = 1
    completed = 0
    actor = Actor(eligibilityDecayActor, learningRateActor, epsilon, policyTable,
                  discountFactorActor)

    if criticType == "neuralnet":
        critic = CriticNeural(
            hiddenLayersDim=hiddenLayersDim,
            eligibilityDecay=eligibilityDecayCritic,
            learningRate=learningRateCritic,
            valueTable=valueTable,
            boardSize=boardSize,
            boardType=boardType,
            discountFactor=discountFactorCritic)
    else:
        print("Using regular critic")
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
        state = world.stateToList()

        chosenAction = actor.ChooseActionByPolicy(world)

        while True:
            reward = world.makeAction(chosenAction)
            nextAction = actor.ChooseActionByPolicy(world)
            nextState = world.stateToList()

            actor.eligibility[str(state) + str(chosenAction)] = 1
            critic.updateTDError(reward, state, nextState)
            critic.eligibility[str(state)] = 1
            TotalError += abs(critic.tdError)
            for SAP in world.getGameLog():

                critic.updateValue(SAP)
                critic.decayEligibility(SAP)

                actor.updatePolicy(SAP, critic.tdError)
                actor.decayEligibility(SAP)

            if chosenAction == None:
                actor.epsilon = actor.epsilon * epsilonDecayRate
                if reward == 10:
                    completed += 1
                break
            chosenAction = nextAction
            state = nextState
            stepsTaken += 1

        print('Episode:', i, 'TotalMeanError', TotalError /
              stepsTaken, "Epsilon:", actor.epsilon, "reward:", reward, 'completedBoards:', completed)

    WriteTables(critic.getValueTable(), actor.getPolicyTable())


if __name__ == "__main__":
    main()
