from graphCreation import createGraph
from graphVisualization import plotGraph
import pandas as pd

if __name__ == "__main__":
    path = "C://users/Riccardo/Desktop/Marzio/Advanced/Project/"
    filename = path + "eeg_data.csv"

    df = pd.read_csv(filename)

    G = createGraph(df.iloc[0:64, 2].values,df.iloc[0:64, 3:-1].values)

    plotGraph(G)
