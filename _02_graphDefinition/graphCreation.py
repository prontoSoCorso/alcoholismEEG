import math
import networkx as nx
import matplotlib.pyplot as plt
import torch


def createGraph():

    # Definire la configurazione degli elettrodi con coordinate precise
    electrode_positions = {
        "FP1": (-1, 3), "FPZ": (0, 3.2), "FP2": (1, 3),
        "AF7": (-1.5, 2.5), "AF3": (-0.5, 2.5), "AFZ": (0, 2.5), "AF4": (0.5, 2.5), "AF8": (1.5, 2.5),
        "F7": (-2, 2), "F5": (-1.5, 2), "F3": (-1, 2), "F1": (-0.5, 2), "FZ": (0, 2), 
        "F2": (0.5, 2), "F4": (1, 2), "F6": (1.5, 2), "F8": (2, 2),
        "FT7": (-2.5, 1.5), "FC5": (-1.5, 1.5), "FC3": (-1, 1.5), "FC1": (-0.5, 1.5), "FCZ": (0, 1.5),
        "FC2": (0.5, 1.5), "FC4": (1, 1.5), "FC6": (1.5, 1.5), "FT8": (2.5, 1.5),
        "T7": (-3, 1), "C5": (-2, 1), "C3": (-1, 1), "C1": (-0.5, 1), "CZ": (0, 1),
        "C2": (0.5, 1), "C4": (1, 1), "C6": (2, 1), "T8": (3, 1),
        "TP7": (-3.5, 0.5), "CP5": (-2.5, 0.5), "CP3": (-1.5, 0.5), "CP1": (-0.5, 0.5), "CPZ": (0, 0.5),
        "CP2": (0.5, 0.5), "CP4": (1.5, 0.5), "CP6": (2.5, 0.5), "TP8": (3.5, 0.5),
        "P9": (-4, 0), "P7": (-3, 0), "P5": (-2, 0), "P3": (-1, 0), "P1": (-0.5, 0), 
        "PZ": (0, 0), "P2": (0.5, 0), "P4": (1, 0), "P6": (2, 0), "P8": (3, 0), "P10": (4, 0),
        "PO7": (-3, -0.5), "PO3": (-1.5, -0.5), "POZ": (0, -0.5), "PO4": (1.5, -0.5), "PO8": (3, -0.5),
        "O1": (-1, -1), "OZ": (0, -1), "O2": (1, -1),
        "IZ": (0, -1.5)
    }

    # Define a threshold to determine significant connections
    threshold = 1  # Adjust as needed based on your data and requirements

    # Create a graph and add significant edges based on the threshold
    G = nx.Graph()

    # Add nodes (electrodes) to the graph
    for electrode, position in electrode_positions.items():
        G.add_node(electrode, pos=position)
        

    for i, electrode1 in enumerate(G.nodes()):
        for j, electrode2 in enumerate(G.nodes()):
            if i < j:
                distance = math.dist(electrode_positions.get(electrode1), electrode_positions.get(electrode2))
                if  distance <= threshold:
                    G.add_edge(electrode1, electrode2, weight = 1 / distance)

    return G



def createGraphWithWeights(weights):

    # Definire la configurazione degli elettrodi con coordinate precise
    electrode_positions = {
        "FP1": (-1, 3), "FPZ": (0, 3.2), "FP2": (1, 3),
        "AF7": (-1.5, 2.5), "AF3": (-0.5, 2.5), "AFZ": (0, 2.5), "AF4": (0.5, 2.5), "AF8": (1.5, 2.5),
        "F7": (-2, 2), "F5": (-1.5, 2), "F3": (-1, 2), "F1": (-0.5, 2), "FZ": (0, 2), 
        "F2": (0.5, 2), "F4": (1, 2), "F6": (1.5, 2), "F8": (2, 2),
        "FT7": (-2.5, 1.5), "FC5": (-1.5, 1.5), "FC3": (-1, 1.5), "FC1": (-0.5, 1.5), "FCZ": (0, 1.5),
        "FC2": (0.5, 1.5), "FC4": (1, 1.5), "FC6": (1.5, 1.5), "FT8": (2.5, 1.5),
        "T7": (-3, 1), "C5": (-2, 1), "C3": (-1, 1), "C1": (-0.5, 1), "CZ": (0, 1),
        "C2": (0.5, 1), "C4": (1, 1), "C6": (2, 1), "T8": (3, 1),
        "TP7": (-3.5, 0.5), "CP5": (-2.5, 0.5), "CP3": (-1.5, 0.5), "CP1": (-0.5, 0.5), "CPZ": (0, 0.5),
        "CP2": (0.5, 0.5), "CP4": (1.5, 0.5), "CP6": (2.5, 0.5), "TP8": (3.5, 0.5),
        "P9": (-4, 0), "P7": (-3, 0), "P5": (-2, 0), "P3": (-1, 0), "P1": (-0.5, 0), 
        "PZ": (0, 0), "P2": (0.5, 0), "P4": (1, 0), "P6": (2, 0), "P8": (3, 0), "P10": (4, 0),
        "PO7": (-3, -0.5), "PO3": (-1.5, -0.5), "POZ": (0, -0.5), "PO4": (1.5, -0.5), "PO8": (3, -0.5),
        "O1": (-1, -1), "OZ": (0, -1), "O2": (1, -1),
        "IZ": (0, -1.5)
    }


    G = nx.Graph()
    for electrode, position in electrode_positions.items():
        G.add_node(electrode, pos=position)

    num_nodes = len(G.nodes)
    weights_matrix = torch.zeros((num_nodes, num_nodes))

    idx = 0
    for i, node_i in enumerate(G.nodes):
        for j, node_j in enumerate(G.nodes):
            if i < j:
                weights_matrix[i, j] = weights[idx].item()
                weights_matrix[j, i] = weights[idx].item()
                idx += 1

    for i, node_i in enumerate(G.nodes):
        for j, node_j in enumerate(G.nodes):
            if i < j and weights_matrix[i, j] > 0:
                G.add_edge(node_i, node_j, weight=weights_matrix[i, j])

    return G

# Visualization function for NX graph or PyTorch tensor
def plotGraph(G):
    # Plot the graph
    plt.figure(figsize=(10, 8))  # Adjust figure size
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
            edge_color='gray', width=2)  # Adjust node size, font size, and edge width
    edge_labels = {(u, v): f"{w:.2f}" for u, v, w in G.edges(data='weight')}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)  # Adjust font size for edge labels
    plt.title('Graph of EEG Electrode Connections Weighted by Correlation Coefficient')
    plt.savefig("graphEEG")
    #plt.show()
