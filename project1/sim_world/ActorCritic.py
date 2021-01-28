import csv
from SimWorld import SimWorld,Action
import VisualizeBoard as visualizer

valueTable = {}#item1 == value item2 = stepsTaken last time state updated
policyTable = {} #item1 == policyValue item2 = stepsTaken last time state updated
episodeRewards = []
stepsTaken = 0
eligibilityDecayValue = 0.9
eligibilityDecayPolicy = 0.9
discountFactor = 0.9
learningRateActor = 0.1
learningRateCritic = 0.1

def GetValue(key: str) -> float:
    if valueTable.get(key, False):
        eligibility = (discountFactor * eligibilityDecayValue) ** (valueTable[key][1] - stepsTaken)
        return (valueTable[key][0], eligibility)
    else:
        return (0,0)

def SetValue(key, value):
    valueTable[key] = (value, GetValue(key)[1])

def SetValueEligibility(key, value):
    valueTable[key] = (GetValue(key)[0], value)

def GetPolicy(key: str) -> float:
    if policyTable.get(key, False):
        eligibility = (discountFactor * eligibilityDecayPolicy) ** (policyTable[key][1] - stepsTaken)
        return (policyTable[key][0], eligibility)
    else:
        return (0,0)

def SetPolicy(key, value):
    policyTable[key] = (value, GetPolicy(key)[1])

def SetPolicyEligibility(key, value):
    policyTable[key] = (GetPolicy(key)[0], value)

def ChooseActionByPolicy(world):
    return world.getLegalActions()[0]#TODO implement policy

def DoEpisodes(episodes, boardSize, stepsTaken):
    #TODO Read Tables
    removePegs = [(0, 0)] #TODO Randomize
    world = SimWorld("triangle", 3, removePegs)
    
    for i in range(episodes):
        reward = 0
        
        print(stepsTaken)
        chosenAction = ChooseActionByPolicy(world)
        while (len(world.getLegalActions()) > 0):
            nextAction = ChooseActionByPolicy(world)# next action
            SetPolicyEligibility(world.stateToHash() + str(chosenAction), stepsTaken)
            
            TDError = reward + discountFactor * GetValue(world.peekAction(nextAction))[0] - GetValue(world.stateToHash)[0]
            SetValueEligibility(world.stateToHash(), stepsTaken)

            for SAP in world.getGameLog():
                value = GetValue(str(SAP[0]))[0]
                eligibilityValue = GetValue(str(SAP[0]))[1]
                SetValue(str(SAP[0]), value + learningRateCritic* TDError* eligibilityValue)
                #skipping pseudocodes eligibility update
                
                policyKey = str(SAP[0]) + str(SAP[1])
                poicyValue = GetPolicy(policyKey)[0]
                eligibilityPolicy = GetPolicy(policyKey)[1]
                SetPolicy(policyKey, poicyValue + learningRateActor* TDError * eligibilityPolicy)
                
            
            reward = world.makeAction(chosenAction) #Make action
            chosenAction = nextAction
            stepsTaken += 1
            print(stepsTaken)

    WriteTables()


def WriteTables(): #TODO Fix eligibilities
    with open('value.csv', 'w') as f:
        for key in valueTable.keys():
            f.write("%s,%s\n"%(key,valueTable[key]))

    with open('policy.csv', 'w') as f:
        for key in policyTable.keys():
            f.write("%s,%s\n"%(key,policyTable[key]))

##if __name__ == "__main__": 

print(stepsTaken)
DoEpisodes(1, 5, stepsTaken)
