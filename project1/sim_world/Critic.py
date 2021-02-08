
class Critic:
    def __init__(self,
                 learningRate=0.1,
                 eligibilityDecay=0.9,
                 valueTable={},
                 discountFactor=0.9
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
        self.tdError = reward + \
            (self.discountFactor * self.getValue(nextState)) - self.getValue(state)

    def updateValue(self, StateActionPair):
        currentEligibility = self.eligibility[StateActionPair.stateHash]
        value = self.getValue(StateActionPair.stateHash)
        self.setValue(StateActionPair.stateHash, value +
                      (self.learningRate * self.tdError * currentEligibility))

    def decayEligibility(self, StateActionPair):
        currentEligibility = self.eligibility[StateActionPair.stateHash]
        self.eligibility[StateActionPair.stateHash] = currentEligibility * \
            self.discountFactor * self.eligibilityDecay
