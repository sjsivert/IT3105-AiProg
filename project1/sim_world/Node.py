class Node():
    # TODO: Add state
    nodeNumber = 0

    def __init__(self) -> None:
        self.location = Node.nodeNumber
        Node.nodeNumber += 1
        self.neighboursDic = {}
        """
        self.neighboursDic = {
            "-1,-1": "1", #up(-1) to left (-1)
            "-1,1": "2", #up (-1) to right (1)
            "0,-1": "3", # sameLine (0) to left(-1)
            "0,1": "5", # sameLine (0) to right(1)
            "1,-1": "8", #down(1) to left(-1)
            "1,1": "9" #down(1) to right(1)
        }
        """

    def __eq__(self, o: object) -> bool:
        return self.location == o.location

    def addNeighbour(self, node, location: str) -> bool:
        # if node in self.neighbours and node != self:
        if node in self.neighboursDic.values() and node != self:
            return False
        self.neighboursDic[location] = node
        return True

    def __str__(self) -> str:
        return str(self.location)

    def __repr__(self) -> str:
        return str(self.location)


class Peg(Node):
    def __init__(self, location, pegValue=1):
        super().__init__()
        self.location = location
        self.neighboursDic = {}
        self.pegValue = pegValue

    def __eq__(self, o: object) -> bool:
        return self.location == o.location

    def addNeighbour(self, node, location: str) -> bool:
        # if node in self.neighbours and node != self:
        if node in self.neighboursDic.values() and node != self:
            return False
        self.neighboursDic[location] = node
        return True

    def addBiDirectionalNeighbour(self, node, location: str) -> bool:
        # if node in self.neighbours and node != self:
        locationList = location.split(',')
        invertedLocation = str(-int(locationList[0]))+ "," + str(-int(locationList[1]))
        if node in self.neighboursDic.values() and node != self:
            return False
        self.neighboursDic[location] = node
        node.addNeighbour(self, invertedLocation)
        return True

    def __str__(self) -> str:
        return str(self.pegValue)

    def __repr__(self) -> str:
        return str(self.pegValue)
