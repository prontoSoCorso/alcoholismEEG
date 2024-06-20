import numpy as np
import matplotlib.pyplot as plt
import umap

def plot_umap(data, labels, params):
    # Define colors for labels
    colors = np.array(['blue', 'green'])

    # UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=params.get('n_neighbors'))
    umap_results = umap_model.fit_transform(data)

    # Plotting
    plt.figure(figsize=(14, 6))

    # UMAP plot
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(umap_results[indices, 0], umap_results[indices, 1], c=colors[label], label=f'Label {label}', alpha=0.6)
    plt.title('UMAP')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()

    plt.show()