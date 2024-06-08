import torch
import torch.nn as nn
import torch_geometric


import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
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


    def forward(self, data):
        
        # Pass data through LSTM
        lstm_out = self.lstm(data)  # Il data è il singolo batch fornito (3D (batch_size, channels, seq_lenght))

        # L'output della LSTM è la feature matrix. Da qui devo creare il pyg data list da fornire come input alla GCN
        pyg_data_list = graphNetworkxToPyg.create_pyg_data_list(self.G, lstm_out)

        # Converto la pyg list in un oggetto batch di pyg
        pyg_batch = torch_geometric.data.Batch.from_data_list(pyg_data_list) 

        # Creazione del tensore batch
        batch_size = lstm_out.size(0)  # Numero di grafi nel batch
        num_nodes = len(self.G.nodes)  # Numero di nodi per grafo
        info_batch = torch.tensor([i for i in range(batch_size) for _ in range(num_nodes)], dtype=torch.long)

        # Pass LSTM output through GCN
        gcn_out = self.GCN(pyg_batch, info_batch)

        # Apply final linear layer
        logits = self.fc(gcn_out)

        return logits


