import csv
from sim_world.SimWorld import SimWorld, Action
# import VisualizeBoard as visualizer
import random


class Actor():
    def __init__(self,
                 eligibilityDecay=0.9,
                 learningRate=0.1,
                 epsilon=0.6,
                 policyTable={},
                 discountFactor=0.9
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
        self.eligibility[SateActionPair.stateHash] = currentEligibility * \
            self.discountFactor * self.eligibilityDecay

    def getPolicyTable(self) -> dict:
        return self.policyTable

    def setPolicy(self, key, value):
        self.policyTable[key] = value

    def ChooseActionByPolicy(self, world):
        actions = world.getLegalActions()
        if len(actions) == 0:  # Safeguard against wrong startingstate
            return None
        if random.uniform(0, 1) > self.epsilon:
            bestIndex = 0
            value = -100000000000000000
            for i in range(len(actions)):
                if self.getPolicy(world.stateToHash() + str(actions[i])) > value:
                    bestIndex = i
                    value = self.getPolicy(
                        world.stateToHash() + str(actions[i]))
            return actions[bestIndex]

        maxRand = len(actions) - 1
        rand = random.randint(0, maxRand)
        return actions[rand]
