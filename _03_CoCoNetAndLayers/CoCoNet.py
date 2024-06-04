import torch
import torch.nn as nn

import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
from _02_graphDefinition import graphNetworkxToPyg
import LSTM, GCN



class CoCoNet(torch.nn.Module):
    def __init__(self, seq_lenght, hidden_size, num_layers, bidirectional, num_classes, dim_lastConvGCN, G):
        super(CoCoNet, self).__init__()
        self.G = G

        self.lstm = LSTM.LSTMnetwork(seq_lenght, hidden_size, num_layers, bidirectional)

        # Output size per layer LSTM, considerando bidirectional
        if bidirectional:
            output_size_lstm = hidden_size * 2
        else:
            output_size_lstm = hidden_size

        self.GCN = GCN.GCN(output_size_lstm)
        self.fc = nn.Linear(dim_lastConvGCN, num_classes)


    def forward(self, data):
        
        # Pass data through LSTM
        lstm_out, _ = self.lstm(data)  # Il data è il singolo batch fornito (3D (batch_size, channels, seq_lenght))

        # L'output della LSTM è la feature matrix. Da qui devo creare il pyg data list da fornire come input alla GCN
        pyg_data_list = graphNetworkxToPyg.create_pyg_data_list(self.G, lstm_out)

        # Pass LSTM output through GCN
        gcn_out = self.GCN(pyg_data_list)

        # Apply final linear layer
        logits = self.fc(gcn_out)
        return logits



