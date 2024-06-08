from graphCreation import createGraph
from graphVisualization import plotGraph

if __name__ == "__main__":

    G = createGraph()
    G.edge.data()
    plotGraph(G)
