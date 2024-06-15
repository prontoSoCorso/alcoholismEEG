import torch
import torch.nn as nn
from torchview import draw_graph


'''
In ingresso la LSTM ha un vettore del tipo: (8, 50, 3)
In questo vettore, il primo valore è la dimensione del batch, ovvero il numero di sequenze passate
Il secondo valore è la lunghezza della singola sequenza. 
Il terzo valore corrisponde al vettore ad ogni singolo timestep. Se avesso ad esempio 3 canali RGB, il valore sarebbe appunto 3

La LSTM restituisce così in output un tensore di dimensione (8,50,5), in cui
8 è sempre il numero di sequenze
5 è il vettore dell'informazione riassuntiva che tiene conto di tutta la sequenza
50 è presente perché per ogni timestep la LSTM calcola le feature per ogni sequenza. E' come avere una matrice 8x5 per i 50 tempi
Proprio per questo alla fine prendo solo l'ultimo (-1) della seconda dimensione con out[:,-1,:] --> come prendere l'ultima matrice all'ultimo tempo delle sequenze


Nel caso degli EEG il problema è che ogni riga della matrice 64x256 deve essere analizzata indipendentemente, ma dalla stessa LSTM, prima di aggiornare i pesi della rete. 
Ogni elettrodo infatti rileva un segnale di cui voglio raccoglierne delle informazioni riassuntive, che saranno della dimensione dell'hidden_size
Dunque, prendo un batch di matrici e vado ad applicare in modo sequenziale la mia LSTM all'i-esima riga di ogni matrice (batch di righe).
Una volta applicata alla singola riga ne salvo l'output e ripeto per le 64 righe.
Fatto questo, ne faccio lo stack e solo a quel punto posso aggiornare i pesi o, come nel nostro caso, passare la matrice allo strato successivo di una rete più complessa
per poi andare ad aggiornare i pesi solo alla fine di tutto.

UPDATE: non è necessario, se si fa la reshape basta passare un unico batch gigante, applicare LSTM e poi fare di nuovo reshape

'''



class LSTMnetwork(nn.Module):
    def __init__(self, seq_length, hidden_size, num_layers, bidirectional):
        super(LSTMnetwork, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=seq_length, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        # x shape: (batch_size, num_channels, seq_length)
        batch_size, num_channels, seq_length = x.size()
        
        # Inizializzo gli hidden states e le cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size*num_channels, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size*num_channels, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x.reshape(batch_size*num_channels, 1, seq_length), (h0, c0))    # input.shape = (1024,1,256),  lstm_out.shape = (1024,1,64)
        lstm_out = lstm_out.reshape(batch_size, num_channels, self.hidden_size)                 # Riottengo forma del tipo (16,64,64)

        
        '''
        # Elaboro ogni canale indipendentemente attraverso la LSTM
        # Per questo faccio il ciclo for andando a prendere x[:,i,:], selezionando così un canale alla volta
        lstm_out = []
        for i in range(num_channels):
            out, _ = self.lstm(x[:,i,:].unsqueeze(1), (h0, c0))  # output shape: (batch_size, 1, hidden_size)
            lstm_out.append(out[:, -1, :])  # Prendo solo (batch_size, hidden_size)

        # Stack all the outputs from the LSTM
        lstm_out = torch.stack(lstm_out)  # shape: (num_channels, batch_size, hidden_size)
        
        # Transpose back to original shape (batch_size, num_channels, hidden_size)
        lstm_out = lstm_out.transpose(0, 1)  # shape: (batch_size, num_channels, hidden_size)
        '''

        return lstm_out

