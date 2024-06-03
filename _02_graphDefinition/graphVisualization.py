import networkx as nx
import matplotlib.pyplot as plt

# Visualization function for NX graph or PyTorch tensor
def plotGraph(G):
    # Plot the graph
    plt.figure(figsize=(12, 10))  # Adjust figure size
    pos = nx.spring_layout(G,
                           seed=42)  # Position nodes using the spring layout algorithm with fixed seed for reproducibility
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold',
            edge_color='gray', width=2)  # Adjust node size, font size, and edge width
    edge_labels = {(u, v): f"{w:.2f}" for u, v, w in G.edges(data='weight')}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)  # Adjust font size for edge labels
    plt.title('Graph of EEG Electrode Connections Weighted by Correlation Coefficient')
    plt.show()