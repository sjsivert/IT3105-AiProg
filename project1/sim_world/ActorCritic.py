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

solvableRemovePegs = {}

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

    def decayEligibility(self, SateActionPair):
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

def GetRandomizedBoard(boardSize, RemovePegs, boardType):
    newWorld = SimWorld(boardType, boardSize)
    newWorld.RemoveRandomPegs(RemovePegs)
    return newWorld

def GetSolvableRemovePegs(boardSize, name):
    solvableInSize = len(solvableRemovePegs[boardSize])
    return solvableRemovePegs[boardSize][min(solvableInSize - 1, name)]

def GetSolvableBoard(boardSize, boardType, name):
    removePegs = GetSolvableRemovePegs(boardType + str(boardSize), name)
    return SimWorld(boardType, boardSize, removePegs)

def updateSolvableStates(boardName, removeLocations):
    if boardName not in solvableRemovePegs.keys():
        solvableRemovePegs[boardName] = []
    notIn = True
    for removeTuples in solvableRemovePegs[boardName]:
        if len(removeTuples) == len(removeLocations):
            inTuple = True
            for removeTuple in removeLocations:
                if removeTuple not in removeTuples:
                    inTuple = False
                    break
            if inTuple:
                notIn = False
    
    if notIn:
        solvableRemovePegs[boardName].append(removeLocations)

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

            actor.eligibility[state + str(chosenAction)] = 1
            critic.updateTDError(reward, state, nextState)
            critic.eligibility[state] = 1
            TotalError += abs(critic.tdError)
            for SAP in world.getGameLog():

                critic.updateValue(SAP)
                critic.decayEligibility(SAP)
                
                actor.updatePolicy(SAP, critic.tdError)
                actor.decayEligibility(SAP)
            
            if reward == 10:
                #print(world.startRemoveLocations, stepsTaken, world.getGameLog()[-1].stateHash)
                updateSolvableStates(boardType + str(boardSize), world.startRemoveLocations)
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
    reward = 0
    while True:
        world.makeAction(chosenAction)
        visualizer.VisualizePegs(
            world.getState(), stepNumber, chosenAction)
        chosenAction = actor.ChooseActionByPolicy(world)
        if chosenAction == None:
            reward = world.makeAction(chosenAction)
            print("EndState:", world._boardState.state, 'reward:', reward)
            break
        stepNumber += 1
    visualizer.GenerateVideo(stepNumber, name)
    return reward

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
    with open('solvable.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            boardsList = []
            for states in row['States'].split('|'):
                statesList = []
                for tuples in states.split('/'):
                    removeTuple = (int(tuples.split(',')[0]),int(tuples.split(',')[1]))
                    statesList.append(removeTuple)
                boardsList.append(statesList)
            solvableRemovePegs[row['Board']] = boardsList
            
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

    with open('solvable.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['Board', 'States'])
        writer.writeheader()
        for key in solvableRemovePegs.keys():
            solvableSatesString = ''
            for states in solvableRemovePegs[key]:
                for tuples in states:
                    for item in tuples:
                        solvableSatesString += str(item) + ','
                    solvableSatesString = solvableSatesString[:-1]
                    solvableSatesString += '/' 
                solvableSatesString = solvableSatesString[:-1]
                solvableSatesString += '|' 
            solvableSatesString = solvableSatesString[:-1]
            writer.writerow({'Board': key, 'States': solvableSatesString})
ReadTables()
DoEpisodes(5000, 4, 2, 'diamond')
DoEpisodes(5000, 4, 2, 'diamond', 0, 0.04)

totalReward = 0
for i in range(len(solvableRemovePegs['diamond4'])):
    print (i + 1, "/", len(solvableRemovePegs['diamond4']))

    totalReward += TestModel(4, 2, 'diamond', i)

print ((totalReward / 10 )/ len(solvableRemovePegs['diamond4']))