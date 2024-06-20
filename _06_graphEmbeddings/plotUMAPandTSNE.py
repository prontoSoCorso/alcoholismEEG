# UMAP and t-SNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def plot_graph_embeddings(data, labels):
    # Define colors for labels
    colors = np.array(['blue', 'green'])

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    tsne_results = tsne.fit_transform(data)

    # UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_results = umap_model.fit_transform(data)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # t-SNE plot
    for label in np.unique(labels):
        indices = labels == label
        ax1.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=colors[label], label=f'Label {label}', alpha=0.6)
    ax1.set_title('t-SNE')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend()

    # UMAP plot
    for label in np.unique(labels):
        indices = labels == label
        ax2.scatter(umap_results[indices, 0], umap_results[indices, 1], c=colors[label], label=f'Label {label}', alpha=0.6)
    ax2.set_title('UMAP')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend()

    plt.show()

