
class Node:
    todo = True
    # TODO: Strongly typed?
    neighbours: [] = []

def createHexGrid (size: int) -> list:
    grid = []
    for i in range(size):
        for y in range(i+1):
            grid.append([Node()])

    return grid
