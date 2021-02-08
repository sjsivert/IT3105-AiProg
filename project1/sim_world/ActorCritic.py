import csv
from SimWorld import SimWorld, Action
import VisualizeBoard as visualizer
import random
from Actor import Actor
from Critic import Critic


episodeRewards = []
discountFactor = 0.9

solvableRemovePegs = {}

def GetRandomizedBoard(boardSize, RemovePegs, boardType):
    newWorld = SimWorld(boardType, boardSize)
    newWorld.RemoveRandomPegs(RemovePegs)
    return newWorld

def GetSolvableBoard(boardSize, boardType, index):
    removePegs = solvableRemovePegs[boardType + str(boardSize)][index]
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

def TestModel(boardSize, maxRemovePegs, boardType, index):

    _ , actorTable = ReadTables()
    actor = Actor(0.9, 0.1, 0, actorTable)
    stepNumber = 0
    #world = GetRandomizedBoard(boardSize, maxRemovePegs, boardType)
    world = GetSolvableBoard(boardSize, boardType, index)
    chosenAction = actor.ChooseActionByPolicy(world)

    visualizer.VisualizePegs(world.getState(), stepNumber)
    reward = 0
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
    visualizer.GenerateVideo(stepNumber, index)
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