

class SimWorld():
    def __init__(self) -> None:
        pass
        # self.actionListLog
        # self.boardState

    def getLegalActions(self) -> [()]:
        pass

    def makeAction(self, action) -> bool:
        pass

    def toHash(self) -> str:
        pass

    def getReward(self) -> int:
        pass


class Action():
    def __init__(self, moveTo: (), moveFrom: (), jumpOver: ()) -> None:
        self.moveFrom = moveFrom
        self.jumpOver = jumpOver
        self.moveTo = moveTo

    def __repr__(self) -> [(), (), ()]:
        return [self.moveFrom, self.jumpOver, self.moveTo]

    def __str__(self) -> str:
        return str([self.moveFrom, self.moveTo])
