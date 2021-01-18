class Node():
    # TODO: Add state
    nodeNumber = 0

    def __init__(self, location: int = nodeNumber) -> None:
        self.neighbours: list = []
        self.location = Node.nodeNumber
        Node.nodeNumber += 1
        self.neighboursDic = {}
        """
        self.neighboursDic = {
            "-1-1": "1", #up(-1) to left (-1)
            "-11": "2", #up (-1) to right (1)
            "0-1": "3", # sameLine (0) to left(-1)
            "01": "5", # sameLine (0) to right(1)
            "1-1": "8", #down(1) to left(-1)
            "11": "9" #down(1) to right(1)
        }
        """

    def __eq__(self, o: object) -> bool:
        return self.location == o.location

    def addNeighbour(self,  node, location: str) -> bool:
        # if node in self.neighbours and node != self:
        if node.location == 7:
            print(node.neighbours)
        if node in self.neighboursDic.values():
            return False
        self.neighboursDic[location] = node
        return True

    def __str__(self) -> str:
        return str(self.location)

    def __repr__(self) -> str:
        return str(self.location)
