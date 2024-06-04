import torch_geometric.data as pyg_data



def networkx_to_pyg_classification(G, feature_matrix):
    # Extract node information
    node_ids = list(G.nodes)

    # Check compatibility of features with nodes
    if len(feature_matrix) != len(node_ids):
        raise ValueError("Number of features must match the number of nodes")

    # Create PyG data object
    data = pyg_data.Data(x=feature_matrix, edge_index=G.edges.data())
    return data



def create_pyg_data_list(G, feature_matrix):
    pyg_data_list = []

    for i in range(feature_matrix.shape[0]):
        pyg_data = networkx_to_pyg_classification(G, feature_matrix[i, :, :].squeeze(0))
        pyg_data_list.append(pyg_data)

    return pyg_data_list

