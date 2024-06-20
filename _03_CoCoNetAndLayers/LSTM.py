import torch
import torch.nn as nn

'''
Nella situazione classica, la LSTM ha in ingresso un batch di matrici, dunque una dimensione del tipo:
(batch_size, num_channels, seq_length)
In uscita avrò un tensore di dimensione:
(batch_size, num_channels, hidden_size) --> hidden_size*2 se BiLSTM

Nel caso degli EEG il problema è che ogni riga della matrice 64x256 deve essere analizzata indipendentemente, ma dalla stessa LSTM, prima di aggiornare i pesi della rete. 
Ogni elettrodo infatti rileva un segnale di cui voglio raccoglierne delle informazioni riassuntive, che saranno della dimensione dell'hidden_size

La prima soluzione, non ottima, è quella di applicare in modo sequenziale la LSTM all'i-esima riga di ogni matrice (batch di righe).
Una volta applicata alla singola riga ne salvo l'output e ripeto per le 64 righe.
Fatto questo, ne faccio lo stack e solo a quel punto posso aggiornare i pesi o, come nel nostro caso, passare la matrice allo strato successivo di una rete più complessa
per poi andare ad aggiornare i pesi solo alla fine di tutto.

Metodo più furbo: faccio la reshape, così passando un unico batch gigante (batch*num_channels, 1, seq_length)
poi applico la LSTM e poi faccio di nuovo un reshape

Nel nostro caso la situazione è ancora diversa: il batch è di pazienti, quindi trattiamo un tensore 4D, che verrà passato
alla LSTM come un 3D (simile a prima) e poi dovrà essere ritornato come uno 4D
'''

class LSTMnetwork(nn.Module):
    def __init__(self, seq_length, hidden_size, num_layers, bidirectional):
        super(LSTMnetwork, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=seq_length, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, num_trials):
        # x shape: (batch_size, num_channels, seq_length)
        batch_size, num_max_trials, num_channels, seq_length = x.size()

        patient_list = []
        for i in range(batch_size):
            patient_list.append(x[i, :num_trials[i], :, :])
        # Concatenate tensors along the first dimension (depth)
        patient_list = torch.cat(patient_list, dim=0)
        patient_list = patient_list.reshape(sum(num_trials)*num_channels, 1, seq_length)

        # Output size per layer LSTM, considerando bidirectional
        if self.bidirectional:
            output_hidden_size = self.hidden_size * 2
        else:
            output_hidden_size = self.hidden_size

        # Inizializzo gli hidden states e le cell states
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), sum(num_trials)*num_channels, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), sum(num_trials)*num_channels, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(patient_list, (h0, c0))    # input.shape = (sum(num_trials)*num_channels,1,256),  lstm_out.shape = (sum(num_trials)*num_channels,1,64)
        
        # Crea un batch con padding
        padded_batch = torch.zeros((batch_size, num_max_trials, num_channels, output_hidden_size))

        for i in range(batch_size):
            depth = num_trials[i]
            start_index = sum(num_trials[0:i])*num_channels
            end_index = sum(num_trials[0:i+1])*num_channels
            padded_batch[i, :depth, :, :] = lstm_out[start_index:end_index, :, :].reshape(depth, num_channels, output_hidden_size)

        lstm_out = padded_batch     # Riottengo forma del tipo (4, num_max_trials, 64, 64)

        return lstm_out