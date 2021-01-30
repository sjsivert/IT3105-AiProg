import csv
from SimWorld import SimWorld,Action
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

    maxRand = len(actions) -1
    rand = random.randint(0, maxRand)
    return actions[rand]

def GetRemovePegs(boardSize, it):
    List = []
    it = random.randint(1, it)
    for i in range(it):
        x = random.randint(0, boardSize -1)
        t = (x, random.randint(0, x) )
        if t not in List:
            List.append(t)
    return List

def GetRandomizedBoard(boardSize, maxRemovePegs, boardType):
    removePegs = GetRemovePegs(boardSize, maxRemovePegs)
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
            
            TDError = reward + (discountFactor * GetValue(world.stateToHash())) - GetValue(state)
            eligibilityValueDict[state] = 1
            TotalError += abs(TDError)
            for SAP in world.getGameLog():
                value = GetValue(SAP[0])
                eligibilityValue = eligibilityValueDict[SAP[0]]
                SetValue(SAP[0], value + (learningRateCritic* TDError* eligibilityValue))
                
                eligibilityValueDict[SAP[0]] = eligibilityValue * discountFactor * eligibilityDecayValue
                
                policyKey = str(SAP[0]) + str(SAP[1])
                poicyValue = GetPolicy(policyKey)
                eligibilityPolicy = eligibilityPolicyDict[policyKey]
                SetPolicy(policyKey, poicyValue + (learningRateActor* TDError * eligibilityPolicy))
                
                eligibilityPolicyDict[SAP[0]] = eligibilityPolicy * discountFactor * eligibilityDecayPolicy
            
            if chosenAction == None:
                break
            chosenAction = nextAction
            state = nextState
            stepsTaken += 1
        print('Episode:' , i, 'MeanError', TotalError / stepsTaken)
        
    WriteTables()

def TestModel(boardSize, maxRemovePegs, boardType):
    global epsylon
    epsylon = 0
    ReadTables()
    stepNumber = 0
    world = GetRandomizedBoard(boardSize, maxRemovePegs, boardType)
    chosenAction = ChooseActionByPolicy(world)
    
    visualizer.VisualizePegs(world.boardState.state, stepNumber)
    while True:
        world.makeAction(chosenAction)
        visualizer.VisualizePegs(world.boardState.state, stepNumber, chosenAction)
        chosenAction = ChooseActionByPolicy(world)
        if chosenAction == None:
            break
        stepNumber+=1
    visualizer.GenerateVideo(stepNumber)
def ReadTables():
    with open('value.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            valueTable[row['Value']] = float(row['Eligibility'])
    with open('policy.csv', mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            policyTable[row['Policy']] = float(row['Eligibility'])

def WriteTables():
    with open('value.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['Value', 'Eligibility'])
        writer.writeheader()
        for key in valueTable.keys():
            writer.writerow({'Value': key, 'Eligibility': valueTable[key]})

    with open('policy.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['Policy', 'Eligibility'])
        writer.writeheader()
        for key in policyTable.keys():
            writer.writerow({'Policy': key, 'Eligibility': policyTable[key]})




TestModel(4, 4, 'triangle')