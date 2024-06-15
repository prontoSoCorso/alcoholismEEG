import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
from torch_geometric.nn.pool import global_mean_pool

import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "alcoholismEEG"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "alcoholismEEG":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from config import utils


class GCN(torch.nn.Module):
  def __init__(self, num_features):
    super(GCN, self).__init__()
    self.conv1 = GCNConv(num_features, utils.dim_firstConvGCN)  # First GCN layer with 16 hidden features
    self.conv1d_1 = nn.Conv1d(1,1,kernel_size=utils.kernel_size_1, padding=utils.padding_size_1)  # First CNN 1d layer
    self.conv2 = GCNConv(utils.dim_firstConvGCN, utils.dim_lastConvGCN)  # Second GCN layer with output as number of classes
    self.conv1d_2 = nn.Conv1d(1,1,kernel_size=utils.kernel_size_2, padding=utils.padding_size_2) # Second CNN 1d layer
    self.pooling = global_mean_pool


  def forward(self, data, info_batch):  # pyg batch as input
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

    h = self.conv1(x, edge_index, edge_weight)
    h = torch.nn.functional.relu(h)  # Apply ReLU activation
    h = self.conv1d_1(h.reshape(h.size(0), 1, utils.dim_firstConvGCN)).squeeze(1)
    h = torch.nn.functional.relu(h)  # Apply ReLU activation
    

    h = self.conv2(h, edge_index, edge_weight )
    h = torch.nn.functional.relu(h)  # Apply ReLU activation
    h = self.conv1d_2(h.reshape(h.size(0), 1, utils.dim_lastConvGCN)).squeeze(1)
    h = torch.nn.functional.relu(h)  # Apply ReLU activation

    # Apply pooling to get graph embedding
    h = self.pooling(h, info_batch)  # Pass batch information for correct pooling
    return h
