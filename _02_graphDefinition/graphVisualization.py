import networkx as nx
import matplotlib.pyplot as plt

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
    plt.show()
