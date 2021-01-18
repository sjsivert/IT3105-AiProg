import networkx as nx

import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import plotly.basedatatypes as base
from HexBoard import createHexGrid
from HexBoard import Node
import plotly.graph_objects as go

# plt.
from plotly.graph_objs import Scatter
from plotly.subplots import make_subplots


def plotTree(node: Node, G, drawnList: list) -> None:
    if node not in drawnList:
        G.add_node(node.location)
        drawnList.append(node)
        for child in node.neighboursDic.values():
            G.add_edge(node.location, child.location)
            plotTree(child, G, drawnList)
    return G


def draw_figure(rootNode: Node):
    #    dirname = os.pak1th.dirname(__file__)
    #    filename = os.path.join(dirname, "test.jpg")
    # font1 = ImageFont.load_default()
    # G = nx.from_pandas_dataframe(df, 'from', 'to', create_uding=nx.Graph())
    ##G = nx.Graph([1,2, {"color": "yellow"}])
    G = nx.Graph()
    rootNode = createHexGrid(3)[0][0]
    G = plotTree(rootNode, G, [])

    # G.add_nodes_from()
    # G[2]['color'] = "red"
    plt.subplot(121)
    nx.draw(
        G,
        with_labels=True,
        #node_color={"blue", "red"},
        font_weight='bold')
    plt.show()


if __name__ == "__main__":
    board = createHexGrid(3)
    draw_figure(
        board[0][0]
    )
