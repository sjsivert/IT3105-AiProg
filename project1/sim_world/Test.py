
from GenerateBoard import HexBoard
from GenerateBoard import BoardState
from GenerateBoard import Boardtype
import VisualizeBoard as visualizer
import simWorldActionManager as actionManager
from matplotlib.animation import FuncAnimation

removePegs = [(0, 0), (1, 0)]
board = HexBoard(Boardtype.triangle,10, removePegs)
state = BoardState(board)

visualizer.VisualizePegs(state.state)
actions = actionManager.GetLegalMoves(state.state)
actionManager.makeMove(state.state, actions[0])


visualizer.VisualizePegs(state.state, actions[0])