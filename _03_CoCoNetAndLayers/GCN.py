import torch
from torch_geometric.nn import GCNConv

import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
from config import utils


class GCN(torch.nn.Module):
  def __init__(self, num_features):
    super(GCN, self).__init__()
    self.conv1 = GCNConv(num_features, utils.dim_firstConvGCN)  # First GCN layer with 16 hidden features
    self.conv2 = GCNConv(utils.dim_firstConvGCN, utils.dim_lastConvGCN)  # Second GCN layer with output as number of classes
  


  def forward(self, data):  # pyg data list as input
    x, edge_index = data.x, data.edge_index
    h = self.conv1(x, edge_index)
    h = torch.nn.functional.relu(h)  # Apply ReLU activation
    h = self.conv2(h, edge_index)

    return torch.nn.functional.log_softmax(x, dim=1)  # Softmax for classification



