
import GenerateBoard as generator

b = generator.GenerateTriangle(4)
b = generator.removePegs([(0,0), (1,0)], b)
state = generator.BoardToNodes(b) 
print("-------------")  
print (state)  
def GetLegalMoves(state):
    actionList = []
    #TODO Change?
    flatNodeList = []
    for sublist in state:
        for item in sublist:
            flatNodeList.append(item)
    for node in flatNodeList:
        for direction, neighboar in node.neighboursDic.items():
            if neighboar.pegValue == 1 and direction in neighboar.neighboursDic:
                if neighboar.neighboursDic[direction].pegValue == 0:
                    jumpFrom = node.location
                    jumpTo = neighboar.neighboursDic[direction].location
                    jumpOver = neighboar.location
                    actionList.append([(jumpFrom),(jumpOver),(jumpTo)])
    return actionList
#TODO make pegMethod: set value
def makeMove(state, action):
    state[action[0][0]][action[0][1]].pegValue = 0
    state[action[1][0]][action[1][1]].pegValue = 0
    state[action[2][0]][action[2][1]].pegValue = 1

actions = GetLegalMoves(state)
print(actions)

makeMove(state, actions[0])
print(state)


actions = GetLegalMoves(state)
print(actions)

makeMove(state, actions[0])
print(state)

actions = GetLegalMoves(state)
print(actions)

makeMove(state, actions[0])
print(state)

actions = GetLegalMoves(state)
print(actions)

makeMove(state, actions[0])
print(state)

actions = GetLegalMoves(state)
print(actions)

makeMove(state, actions[0])
print(state)

actions = GetLegalMoves(state)
print(actions)
