
class Node:
    todo = True
    # TODO: Strongly typed?

    def __init__(self, location: int = 1) -> None:
        self.neighbours: list = []
        self.location = location

    def __eq__(self, o: object) -> bool:
        return True

    def addNeighbour(self,  node) -> bool:
        if node in self.neighbours:
            return False
        self.neighbours.append(node)
        print(self.neighbours)
        return True

    def __str__(self) -> str:
        return "node"

    def __repr__(self) -> str:
        return str(self.location)


def createHexGrid(size: int) -> list:
    grid = []
    for i in range(size):
        node = Node(i)
        if i != 0 and (i < size):
            # Add neighbors to node above
            node.addNeighbour(grid[i-1])
            grid[-1][0].addNeighbour(node)
        grid.append([node])
        for y in range(i):
            node2 = Node(i)
            grid[i].append(node2)

            # Add neighbour to node above
            node.addNeighbour(grid[i-1])
            grid[-1][0].addNeighbour(node)

            if y != 1 and y < size:
                # Add neighbour to node behind
                node2.addNeighbour(grid[i][y-1])
                grid[i][y-1].addNeighbour(grid[i][y-1])

    return grid


if __name__ == "__main__":
    t = createHexGrid(3)
    print(t)
