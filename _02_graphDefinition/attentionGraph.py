import torch

def average_subtensors(tensor_list, m):
    # List to hold all [m, 1] subtensors
    all_subtensors = []

    # Iterate through each tensor in the list
    for tensor in tensor_list:
        # Ensure the tensor is 2D with shape [n, 1]
        assert tensor.dim() == 2 and tensor.size(1) == 1

        # Split the tensor into [m, 1] subtensors
        n = tensor.size(0)
        assert n % m == 0, "n should be a multiple of m"
        
        subtensors = tensor.view(n // m, m, 1)
        all_subtensors.append(subtensors)

    # Concatenate all subtensors into one tensor of shape [-1, m, 1]
    all_subtensors = torch.cat(all_subtensors, dim=0)

    # Compute the average of all subtensors
    average_tensor = all_subtensors.mean(dim=0)

    return average_tensor

"""
# Example usage:
tensor_list = [torch.randn(12, 1), torch.randn(24, 1), torch.randn(36, 1)]
m = 6
average_tensor = average_subtensors(tensor_list, m)
print(average_tensor)
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_weight_difference(graph, new_weights, file_name, node_importance = None):
    max_weight = max([graph[u][v]['weight'] for u, v in graph.edges()])
    min_weight = min([graph[u][v]['weight'] for u, v in graph.edges()])

    # Compare edge weights and assign colors
    edge_colors = []
    count = 0
    t = 0.1
    for u, v in graph.edges():

        old_weight = (graph[u][v]['weight']-min_weight)/(max_weight-min_weight)
        new_weight = (new_weights[count]-min(new_weights))/(max(new_weights)-min(new_weights))
        
        if old_weight+t < new_weight:
            edge_colors.append('green')  # Weight increased
        elif old_weight-t > new_weight:
            edge_colors.append('red')    # Weight decreased
        else:
            edge_colors.append('gray')   # Weight stayed the same

        graph[u][v]['weight'] = abs(old_weight-new_weight)
            
        count = count+1

    """
    # Assign colors to nodes based on importance
    node_colors = []
    max_importance = max(node_importance.values())
    min_importance = min(node_importance.values())
    norm = mcolors.Normalize(vmin=min_importance, vmax=max_importance)
    cmap = cm.viridis
    
    for node in unified_graph.nodes():
        importance = node_importance.get(node, 0)
        # Normalize importance and get the corresponding color
        node_colors.append(cmap(norm(importance)))
    """

    # Plot the unified graph
    plt.figure(figsize=(10, 8))  # Adjust figure size
    pos = nx.get_node_attributes(graph, 'pos')
    #pos = nx.spring_layout(unified_graph)  # Layout for visualization
    #nx.draw(unified_graph, pos, edge_color=edge_colors, node_color=node_colors, with_labels=True, node_size=500)
    nx.draw(G=graph, pos=pos, edge_color=edge_colors, with_labels=True, node_size=500)
    #edge_labels = {(u, v): f"{w:.2f}" for u, v, w in graph.edges(data='weight')}
    #nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)  # Adjust font size for edge labels

    # Create a colorbar as a legend
    #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])  # Only needed for the colorbar to work
    #cbar = plt.colorbar(sm)
    #cbar.set_label('Node Importance')

    plt.savefig("attGraphS1_" + file_name)

"""
# Example usage
# Create two graphs with the same structure but different edge weights
G1 = nx.Graph()
G1.add_edge(1, 2, weight=1)
G1.add_edge(2, 3, weight=2)
G1.add_edge(3, 1, weight=3)

G2 = nx.Graph()
G2.add_edge(1, 2, weight=1)
G2.add_edge(2, 3, weight=0.1)
G2.add_edge(3, 1, weight=0.1)

new_weights = [G2[u][v]['weight'] for u, v in G2.edges()]

# Define node importance
node_importance = {1: 0.1, 2: 0.5, 3: 0.9}

plot_weight_difference(G1, new_weights, node_importance)

"""