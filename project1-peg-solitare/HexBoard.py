
class Node:
    todo = True
    # TODO: Strongly typed?

    def __init__(self, location: int = 1) -> None:
        self.neighbours: list = []
        self.location = location

    def __eq__(self, o: object) -> bool:
        return True

    def addNeighbour(self,  node) -> bool:
        if node in self.neighbours and node != self:
            return False
        self.neighbours.append(node)
        return True

    def __str__(self) -> str:
        return "node"

    def __repr__(self) -> str:
        return str(self.location)


def createChildren(grid, parents: list, numberOfChildren: int, deapthLimit: int):
    childrens = []
    for index in range(numberOfChildren - 1):
        child = Node()
        # Add parents as neighboars
        for parent in parents:
            child.addNeighbour(parent)
            parent.addNeighbour(child)
        for previousMadeChildren in childrens:
            child.addNeighbour(previousMadeChildren)
        childrens.append(child)

    if deapthLimit == 0:
        return grid.append(childrens)
    else:
        return grid.append(createChildren(grid, childrens, numberOfChildren+1, deapthLimit-1))


def createHexGrid(size: int) -> list:
    grid = []
    # Create root node
    root = Node()
    grid.append([root])
    grid.append(
        createChildren([root], 2, 3)
    )
    return grid


"""     for i in range(size):
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
 """


if __name__ == "__main__":
    t = createHexGrid(3)
    print(t)
