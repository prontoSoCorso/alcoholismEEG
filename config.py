''' Configuration file for the project'''
import torch

class user_paths:
    # Computer Lorenzo
    path_alcoholismEEG_data = "C:/Users/loren/Documents/Data/eegData/eeg_full/"
    output_path_csv = "C:/Users/loren/Documents/Data/eegData/eeg_data.csv"

    # Computer Marzio



class utils:
    # generics
    seed = 2024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # LSTM params
    seq_lenght =  256
    hidden_size = 64
    num_layers = 2
    bidirectional = False
    num_classes = 2
   
    # GCN params
    dim_firstConvGCN = 32
    dim_lastConvGCN = 8

    # CoCoNet params
    batch_size = 16
    num_epochs = 10
