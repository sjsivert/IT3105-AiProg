import csv
from SimWorld import SimWorld, Action
import VisualizeBoard as visualizer
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

class Actor():
    def __init__(self,
        eligibilityDecay = 0.9,
        learningRate = 0.1,
        epsilon = 0.6,
        policyTable = {},
        discountFactor = 0.9
    ) -> None:
        self.policyTable = policyTable
        self.eligibility = {}
        self.eligibilityDecay = eligibilityDecay
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.discountFactor = discountFactor

    def resetEligibility(self):
        self.eligibility = {}
    
    def getPolicy(self, key: str) -> float:
        if self.policyTable.get(key, False):
            return self.policyTable[key]
        else:
            return 0

    def updatePolicy(self, SateActionPair, tdError):
        policyKey = str(SateActionPair.stateHash) + str(SateActionPair.action)
        poicyValue = self.getPolicy(policyKey)
        currentEligibility = self.eligibility[policyKey]
        policyChange = (self.learningRate * tdError * currentEligibility)
        self.setPolicy(policyKey, poicyValue + policyChange)

    def updateEligibility(self, SateActionPair):
        policyKey = str(SateActionPair.stateHash) + str(SateActionPair.action)
        currentEligibility = self.eligibility[policyKey]
        self.eligibility[SateActionPair.stateHash] = currentEligibility * self.discountFactor * self.eligibilityDecay
       
    def getPolicyTable(self) -> dict:
        return self.policyTable

    def setPolicy(self, key, value):
        self.policyTable[key] = value

    def ChooseActionByPolicy(self, world):
        actions = world.getLegalActions()
        if len(actions) == 0: # Safeguard against wrong startingstate
            return None
        if random.uniform(0, 1) > self.epsilon:
            bestIndex = 0
            value = -100000000000000000
            for i in range(len(actions)):
                if self.getPolicy(world.stateToHash() + str(actions[i])) > value:
                    bestIndex = i
                    value = self.getPolicy(world.stateToHash() + str(actions[i]))
            return actions[bestIndex]

        maxRand = len(actions) - 1
        rand = random.randint(0, maxRand)
        return actions[rand]
    
    

class Critic: 
    def __init__(self,
        learningRate = 0.1,
        eligibilityDecay = 0.9,
        valueTable = {},
        discountFactor = 0.9
    ) -> None:
        self.valueTable = valueTable
        self.eligibilityDecay = eligibilityDecay
        self.eligibility = {}
        self.tdError = 0
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        
    def resetEligibility(self):
        self.eligibility = {}

    def getValueTable(self) -> dict:
        return self.valueTable

    def getValue(self, key: str) -> float:
        if self.valueTable.get(key, False):
            return self.valueTable[key]
        else:
            return 0

    def setValue(self, key, value):
        self.valueTable[key] = value

    def updateTDError(self, reward, state, nextState):
        self.tdError = reward + (self.discountFactor * self.getValue(nextState)) - self.getValue(state)

    def updateValue(self, StateActionPair):
        currentEligibility = self.eligibility[StateActionPair.stateHash]
        value = self.getValue(StateActionPair.stateHash)
        self.setValue(StateActionPair.stateHash, value + (self.learningRate * self.tdError * currentEligibility))

    def decayEligibility(self, StateActionPair):
        currentEligibility = self.eligibility[StateActionPair.stateHash]
        self.eligibility[StateActionPair.stateHash] = currentEligibility * self.discountFactor * self.eligibilityDecay

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


def DoEpisodes(episodes, boardSize, maxRemovePegs, boardType, epsilon = 0.6, learningRate = 0.1, policyTable = {}, valueTable = {}):
    TotalError = 0
    stepsTaken = 1

    actor = Actor(0.9, learningRate, epsilon, policyTable)    
    critic = Critic(0.9, learningRate, valueTable)    

    for i in range(episodes):
        world = GetRandomizedBoard(boardSize, maxRemovePegs, boardType)
        
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

            actor.eligibility[state + str(chosenAction)] = 1 # Set eligibility to 1 in given state
            critic.updateTDError(reward, state, nextState)
            critic.eligibility[state] = 1
            TotalError += abs(critic.tdError)
            for SAP in world.getGameLog():

                critic.updateValue(SAP)
                critic.decayEligibility(SAP)
                
                actor.updatePolicy(SAP, critic.tdError)
                actor.updateEligibility(SAP)
            
            if chosenAction == None:
                break
            chosenAction = nextAction
            state = nextState
            stepsTaken += 1
            
        print('Episode:', i, 'MeanError', TotalError / stepsTaken)

    WriteTables(critic.getValueTable(), actor.getPolicyTable())


def TestModel(boardSize, maxRemovePegs, boardType, name):

    critricTable , actorTable= ReadTables()
    actor = Actor(0.9, 0.1, 0, actorTable)    
    critic = Critic(0.9, 0.1, critricTable)  
    stepNumber = 0
    #world = GetRandomizedBoard(boardSize, maxRemovePegs, boardType)
    world = GetSolvableBoard(boardSize, boardType, name)
    chosenAction = actor.ChooseActionByPolicy(world)

    visualizer.VisualizePegs(world.getState(), stepNumber)
    while True:
        world.makeAction(chosenAction)
        visualizer.VisualizePegs(
            world.getState(), stepNumber, chosenAction)
        chosenAction = actor.ChooseActionByPolicy(world)
        if chosenAction == None:
            break
        stepNumber += 1
    print(world.getGameLog()[0].stateHash)
    visualizer.GenerateVideo(stepNumber, name)


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


DoEpisodes(5000, 4, 4, 'diamond')
DoEpisodes(5000, 4, 4, 'diamond', 0, 0.04)


TestModel(4, 4, 'diamond', 0)
TestModel(4, 4, 'diamond', 1)
TestModel(4, 4, 'diamond', 2)
TestModel(4, 4, 'diamond', 3)
TestModel(4, 4, 'diamond', 4)
TestModel(4, 4, 'diamond', 5)
TestModel(4, 4, 'diamond', 6)
TestModel(4, 4, 'diamond', 7)

