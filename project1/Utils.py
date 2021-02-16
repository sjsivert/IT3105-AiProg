import csv
from sim_world.SimWorld import SimWorld, Action
import VisualizeBoard as visualizer
from actor_critic.Actor import Actor
from actor_critic.Critic import Critic
import random


episodeRewards = []
discountFactor = 0.9
"""
valueTable = {}
policyTable = {}
eligibilityDecayValue = 0.9
eligibilityDecayPolicy = 0.9
learningRateActor = 0.1
learningRateCritic = 0.1
epsylon = 0.6
"""

solvableRemovePegs = {
    4: [
        [(1, 0)],
        [(2, 0)],
        [(1, 1)],
        [(2, 2)],
        [(3, 1)],
        [(3, 2)],
        [(2, 1)],
        [(2, 0), (3, 0)],
        [(3, 1), (3, 0)],
        [(3, 3), (3, 2)],
        [(3, 3), (2, 2)],
        [(0, 0), (1, 0)],
        [(0, 0), (1, 1)]
    ]
}


def GetRandomisedRemovePegs(boardSize, maxRemovePegs):
    """
        Generates random starting stated
        TODO: Make better
    """
    List = []
    maxRemovePegs = random.randint(1, maxRemovePegs)
    for i in range(maxRemovePegs):
        x = random.randint(0, boardSize - 1)
        t = (x, random.randint(0, x))
        if t not in List:
            List.append(t)
    return List


def GetSolvableRemovePegs(boardSize, name):
    solvableInSize = len(solvableRemovePegs[boardSize])
    return solvableRemovePegs[boardSize][min(solvableInSize - 1, name)]


def GetRandomizedBoard(boardSize, maxRemovePegs, boardType):
    removePegs = GetRandomisedRemovePegs(boardSize, maxRemovePegs)
    return SimWorld(boardType, boardSize, removePegs)


def GetSolvableBoard(boardSize, boardType, name):
    removePegs = GetSolvableRemovePegs(boardSize, name)
    return SimWorld(boardType, boardSize, removePegs)


def TestModel(boardSize, boardType, removePegs):

    _, actorTable = ReadTables()
    actor = Actor(0.9, 0.1, 0, actorTable)
    stepNumber = 0
    world = SimWorld(boardType, boardSize, removePegs)
    chosenAction = actor.ChooseActionByPolicy(world)

    visualizer.VisualizePegs(world.getState(), stepNumber)
    while True:
        world.makeAction(chosenAction)
        visualizer.VisualizePegs(
            world.getState(), stepNumber, chosenAction)
        chosenAction = actor.ChooseActionByPolicy(world)
        if chosenAction == None:
            endstate = str(world._boardState.state)
            reward = world.makeAction(chosenAction)
            print("EndState:", endstate, 'reward:', reward)
            break
        stepNumber += 1
    visualizer.GenerateVideo(stepNumber, boardType + str(boardSize))


def ReadTables():
    values = {}
    policy = {}
    with open('value.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            values[row['stateHash']] = float(row['stateValue'])
    with open('policy.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            policy[row['Policy']] = float(row['Eligibility'])

    return values, policy


def WriteTables(values, policy):
    with open('value.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['stateHash', 'stateValue'])
        writer.writeheader()
        for key in values.keys():
            writer.writerow({'stateHash': key, 'stateValue': values[key]})

    with open('policy.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['Policy', 'Eligibility'])
        writer.writeheader()
        for key in policy.keys():
            writer.writerow({'Policy': key, 'Eligibility': policy[key]})


if __name__ == '__main__':
    #DoEpisodes(5000, 4, 4, 'diamond')
    #DoEpisodes(5000, 4, 4, 'diamond', 0, 0.04)

    TestModel(4, 4, 'diamond', 0)
    TestModel(4, 4, 'diamond', 1)
    TestModel(4, 4, 'diamond', 2)
    TestModel(4, 4, 'diamond', 3)
    TestModel(4, 4, 'diamond', 4)
    TestModel(4, 4, 'diamond', 5)
    TestModel(4, 4, 'diamond', 6)
    TestModel(4, 4, 'diamond', 7)
