Parameters = {
    "useTableCritic": False,
    "useDiamondBoard": False,
    "architecture": [15, 20, 5, 1],
    "showFinalSolution": True,
    "actorLearningRate": 0.75,
    "actorDiscFactor": 0.9,
    "actorTraceDecay": 0.9,
    "criticLearningRate": 0.75,
    "criticDiscFactor": 0.9,
    "criticTraceDecay": 0.8,
    "epsilon": 0.8,
    "epsilonDecay": 0.994,
    "numberOfEpisodes": 800,
    "frameDelay": 500,
}
# examples:

# eDecay = 0.99
# nEpisodes <= 400 ish

# eDecay = 0.994
# nEpisodes >= 900

# 0.997, 1500

DiamondBoard = {
    "emptyStartPins": [6],
    "size": 4
}

TriangleBoard = {
    "emptyStartPins": [0],
    "size": 5
}
