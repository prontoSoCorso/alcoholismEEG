import torch
import torch.nn as nn
import torch_geometric

import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "alcoholismEEG"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "alcoholismEEG":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from _02_graphDefinition import graphNetworkxToPyg
from _03_CoCoNetAndLayers import LSTM, GCN

class CoCoNet(torch.nn.Module):
    def __init__(self, seq_lenght, hidden_size, num_layers, bidirectional, dim_lastConvGCN, G):
        super(CoCoNet, self).__init__()
        self.G = G

        self.lstm = LSTM.LSTMnetwork(seq_lenght, hidden_size, num_layers, bidirectional)

        # Output size per layer LSTM, considerando bidirectional
        if bidirectional:
            output_size_lstm = hidden_size * 2
        else:
            output_size_lstm = hidden_size

        self.GCN = GCN.GCN(output_size_lstm)
        self.fc = nn.Linear(dim_lastConvGCN, 1)

    def forward(self, data, num_trials):
        # Pass data through LSTM
        lstm_out = self.lstm(data, num_trials)  # Il data è il singolo batch fornito (4D (batch_size, num_trials, channels, seq_lenght))

        # L'output della LSTM è la feature matrix. Da qui devo creare il pyg data list da fornire come input alla GCN
        pyg_data_list = graphNetworkxToPyg.create_pyg_data_list(self.G, lstm_out, num_trials)

        # Converto la pyg list in un oggetto batch di pyg
        # A data object describing a batch of graphs as one big (disconnected) graph
        pyg_batch = torch_geometric.data.Batch.from_data_list(pyg_data_list)

        # Creazione del tensore batch
        batch_size = lstm_out.size(0)                               # Numero di pazienti nel batch
        info_batch = torch.tensor([i for i in range(batch_size) for _ in range(len(self.G.nodes)*num_trials[i])], dtype=torch.long)

        # Pass LSTM output through GCN
        gcn_out = self.GCN(pyg_batch, info_batch)

        # Apply final linear layer
        logits = self.fc(gcn_out)

        return logits, gcn_out