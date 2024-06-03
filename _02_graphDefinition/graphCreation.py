import numpy as np
import networkx as nx
from sklearn.metrics import mutual_info_score

def createGraph(electrodes, data):
    num_electrodes, num_samples = data.shape

    # Mutual information
    #mutual_info_matrix = np.zeros((num_electrodes, num_electrodes))
    # Calculate mutual information between each pair of electrodes
    #for i in range(num_electrodes):
    #    for j in range(i + 1, num_electrodes):
    #        mutual_info = mutual_info_score(data[i, :], data[j, :])
    #        mutual_info_matrix[i, j] = mutual_info
    #        mutual_info_matrix[j, i] = mutual_info  # Since mutual information is symmetric

    # Pearson correlation
    correlation_matrix = np.corrcoef(data)

    # Define a threshold to determine significant connections
    threshold = 0.25  # Adjust as needed based on your data and requirements

    # Create a graph and add significant edges based on the threshold
    G = nx.Graph()

    # Add nodes (electrodes) to the graph
    for electrode in electrodes:
        G.add_node(electrode)

    for i, electrode1 in enumerate(G.nodes):
        for j, electrode2 in enumerate(G.nodes):
            if i < j:
                if abs(correlation_matrix[i, j]) >= threshold:
                    G.add_edge(electrode1, electrode2, weight=abs(correlation_matrix[i, j]))

    return G
