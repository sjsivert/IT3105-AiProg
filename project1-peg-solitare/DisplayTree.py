import mode as mode
import networkx as nx

import pandas as pd
from PIL import Image, ImageFont, ImageDraw, ImageDraw2
from PIL import ImageFont
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

def draw_figure(xList: list, yList: list):
#    dirname = os.pak1th.dirname(__file__)
#    filename = os.path.join(dirname, "test.jpg")
    #font1 = ImageFont.load_default()
#G = nx.from_pandas_dataframe(df, 'from', 'to', create_uding=nx.Graph())
    ##G = nx.Graph([1,2, {"color": "yellow"}])
    G = nx.Graph()
    for node in xList:
        G.add_node(node[0].location)
        print(node[0].neighbours )
        for child in node[0].neighbours:
            G.add_node(child.location)
            G.add_edge(node[0].location, child.location)

    #G.add_nodes_from()
   # G[2]['color'] = "red"
    plt.subplot(121)
    nx.draw(
        G,
        with_labels=True,
        #node_color={"blue", "red"},
        font_weight='bold')
    plt.show()
"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x = ["h",2],
            y = [1, 2] ,
            mode='lines',
            line = dict(color='rgb(210,210,210)', width=1),
            hoverinfo = 'none'
        )
    )
    fig.add_trace(go.Scatter(x=[1,1],
                      y=[2,2],
                      mode='markers',
                      name='bla',
                      marker=dict(symbol='circle-dot',
                                    size=18,
                                    color='#6175c1',    #'#DB4551',
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                      text="test",
                      hoverinfo='text',
                      opacity=0.8
                      ))


    fig.show()
"""
"""
    fig = make_subplots()
    fig.add_trace(
        col=  1   ,
        row=  1  ,

        trace=Scatter(x=xList, y=yList)
    )
    """
    #fig.show()

if __name__ == "__main__":
    x = [1,2]
    y = [1, 2]
    board = createHexGrid(3)
    xdic = list(
        map(
            lambda index:
                index[1][index[0]].location
            ,
            enumerate(board)
        )
    )
    print(xdic)
    x = list(range(board.__sizeof__()))
    draw_figure(
        xList = board,
        yList = xdic,
    )

"""
    ImageDraw2.Draw.text(
        self=Image.new(mode= mode.new_module(),
                       
                       # filename =filename,
                       size=(20, 20)),
        xy=(255, 255),
        text="Hello World",
        font=font1
    )
"""
