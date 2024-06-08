import torch_geometric.data as pyg_data
import torch


def networkx_to_pyg_classification(G, feature_matrix):
    # Create a mapping from node names to indices
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

    # Check compatibility of features with nodes
    if len(feature_matrix) != len(node_mapping):
        raise ValueError("Number of features must match the number of nodes")

    '''
    Convertire la lista degli archi in un tensor PyTorch: Usa torch.tensor per convertire la lista degli archi in un tensor. 
    Per PyTorch Geometric, questo tensor deve essere trasposto (usando .t()) e reso contiguo (usando .contiguous()).
    '''
    # Convert edges to the edge_index format required by PyTorch Geometric
    edge_index = torch.tensor(
        [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in G.edges()],
        dtype=torch.long
    ).t().contiguous()

     # Convert edge weights to a tensor
    edge_weights = torch.tensor(
        [G.edges[edge]['weight'] for edge in G.edges()],
        dtype=torch.float
    )
    
    # Crea un oggetto Data di PyTorch Geometric
    data = pyg_data.Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_weights)

    return data



def create_pyg_data_list(G, feature_matrix):
    pyg_data_list = []

    for i in range(feature_matrix.shape[0]):
        pyg_data = networkx_to_pyg_classification(G, feature_matrix[i, :, :].squeeze(0))
        pyg_data_list.append(pyg_data)

    return pyg_data_list

