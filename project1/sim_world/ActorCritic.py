import csv
from SimWorld import SimWorld, Action
import VisualizeBoard as visualizer
import random

valueTable = {}
policyTable = {}
episodeRewards = []
eligibilityDecayValue = 0.9
eligibilityDecayPolicy = 0.9
discountFactor = 0.9
learningRateActor = 0.1
learningRateCritic = 0.1
epsylon = 0.6

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


def GetValue(key: str) -> float:
    if valueTable.get(key, False):
        return valueTable[key]
    else:
        return 0


def SetValue(key, value):
    valueTable[key] = value


def GetPolicy(key: str) -> float:
    if policyTable.get(key, False):
        return policyTable[key]
    else:
        return 0


def SetPolicy(key, value):
    policyTable[key] = value


def ChooseActionByPolicy(world):
    actions = world.getLegalActions()
    if len(actions) == 0:
        return None
    if random.uniform(0, 1) > epsylon:
        bestIndex = 0
        value = -100000000000000000
        for i in range(len(actions)):
            if GetPolicy(world.stateToHash() + str(actions[i])) > value:
                bestIndex = i
                value = GetPolicy(world.stateToHash() + str(actions[i]))
        return actions[bestIndex]

    maxRand = len(actions) - 1
    rand = random.randint(0, maxRand)
    return actions[rand]


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


def DoEpisodes(episodes, boardSize, maxRemovePegs, boardType):
    TotalError = 0
    stepsTaken = 1
    ReadTables()

    for i in range(episodes):
        world = GetRandomizedBoard(boardSize, maxRemovePegs, boardType)

        eligibilityValueDict = {}
        eligibilityPolicyDict = {}
        reward = 0
        chosenAction = None
        state = world.stateToHash()

        if (len(world.getLegalActions()) > 0):
            chosenAction = ChooseActionByPolicy(world)
        while True:

            reward = world.makeAction(chosenAction)

            nextAction = ChooseActionByPolicy(world)
            nextState = world.stateToHash()
            eligibilityPolicyDict[state + str(chosenAction)] = 1

            TDError = reward + \
                (discountFactor * GetValue(world.stateToHash())) - GetValue(state)
            eligibilityValueDict[state] = 1
            TotalError += abs(TDError)

            for SAP in world.getGameLog():
                value = GetValue(SAP.stateHash)
                eligibilityValue = eligibilityValueDict[SAP.stateHash]
                SetValue(SAP.stateHash, value + (learningRateCritic *
                                                 TDError * eligibilityValue))

                eligibilityValueDict[SAP.stateHash] = eligibilityValue * \
                    discountFactor * eligibilityDecayValue

                policyKey = str(SAP.stateHash) + str(SAP.action)
                poicyValue = GetPolicy(policyKey)
                eligibilityPolicy = eligibilityPolicyDict[policyKey]
                SetPolicy(policyKey, poicyValue +
                          (learningRateActor * TDError * eligibilityPolicy))

                eligibilityPolicyDict[SAP.stateHash] = eligibilityPolicy * \
                    discountFactor * eligibilityDecayPolicy

            if chosenAction == None:
                break
            chosenAction = nextAction
            state = nextState
            stepsTaken += 1
        print('Episode:', i, 'MeanError', TotalError / stepsTaken)

    WriteTables()


def TestModel(boardSize, maxRemovePegs, boardType, name):
    global epsylon
    epsylon = 0
    ReadTables()
    stepNumber = 0
    world = GetRandomizedBoard(boardSize, maxRemovePegs, boardType)
    world = GetSolvableBoard(boardSize, boardType, name)
    chosenAction = ChooseActionByPolicy(world)

    visualizer.VisualizePegs(world.getState(), stepNumber)
    while True:
        world.makeAction(chosenAction)
        visualizer.VisualizePegs(
            world.getState(), stepNumber, chosenAction)
        chosenAction = ChooseActionByPolicy(world)
        if chosenAction == None:
            break
        stepNumber += 1
    visualizer.GenerateVideo(stepNumber, name)


def ReadTables():
    with open('value.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            valueTable[row['stateHash']] = float(row['stateValue'])
    with open('policy.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            policyTable[row['Policy']] = float(row['Eligibility'])


def WriteTables():
    with open('value.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['stateHash', 'stateValue'])
        writer.writeheader()
        for key in valueTable.keys():
            writer.writerow({'stateHash': key, 'stateValue': valueTable[key]})

    with open('policy.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['Policy', 'Eligibility'])
        writer.writeheader()
        for key in policyTable.keys():
            writer.writerow({'Policy': key, 'Eligibility': policyTable[key]})


WriteTables()
DoEpisodes(5000, 4, 4, 'diamond')
learningRateActor = 0.04
learningRateCritic = 0.04
epsylon = 0
DoEpisodes(5000, 4, 4, 'diamond')


TestModel(4, 4, 'diamond', 0)
TestModel(4, 4, 'diamond', 1)
TestModel(4, 4, 'diamond', 2)
TestModel(4, 4, 'diamond', 3)
TestModel(4, 4, 'diamond', 4)
TestModel(4, 4, 'diamond', 5)
TestModel(4, 4, 'diamond', 6)
TestModel(4, 4, 'diamond', 7)
