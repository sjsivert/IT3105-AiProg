import random

class RandomAgent ():

    def defaultPolicyFindAction(self, possibleActions, state) -> int:
        print("random Agent - Possible Actions ", possibleActions)
        return(random.choice(possibleActions))