import json
from Node import Node


def createChildren(grid, parents: list, numberOfChildren: int, deapthLimit: int):
    childrens = []
    deapthLimit -= 1
    for index in range(numberOfChildren):
        child = Node()
        # Add parents as neighboars
        if(index == 0):
            parentsToAdd = [parents[index]]
        elif(index == len(parents)):
            parentsToAdd = [parents[index-1]]
        else:
            parentsToAdd = parents[index-1: index+1]
        for parent in parentsToAdd:
            if (index % 2 == 0):
                child.addNeighbour(parent, "-11")
                parent.addNeighbour(child, "1-1")
            elif (index % 2 == 1 or parentsToAdd.index(parent) == 1):
                child.addNeighbour(parent, "-1-1")
                parent.addNeighbour(child, "11")

        for previousMadeChildren in childrens[index-1:index+1]:
            child.addNeighbour(previousMadeChildren, "0-1")
            previousMadeChildren.addNeighbour(child, "01")
        childrens.append(child)

    if deapthLimit == 0:
        return childrens
    else:
        grid.append(childrens)
        createChildren(grid,
                       childrens,
                       numberOfChildren + 1,
                       deapthLimit)


def createHexGrid(size: int) -> list:
    grid = []
    # Create root node
    root = Node()
    grid.append([root])
    createChildren(
        grid=grid,
        parents=[root],
        numberOfChildren=2,
        deapthLimit=size)
    return grid


def printTree(node, printed: list) -> None:
    if node not in printed:
        print(node)
        printed.append(node)
        for child in node.neighbours:
            printTree(child, printed)


if __name__ == "__main__":
    t = createHexGrid(4)
    print(t)
    printTree(t[0][0], [])

    # Check Neighboard
    print(
        t[0][0].neighboursDic
    )
    print(
        t[1][0].neighboursDic
    )
    print(
        t[2][1].neighboursDic
    )
    print(
        t[3][1].neighboursDic
    )
