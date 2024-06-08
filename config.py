''' Configuration file for the project'''
import torch

class user_paths:
    # Computer Lorenzo
    path_alcoholismEEG_data = "C:/Users/loren/Documents/Data/eegData/eeg_full/"
    output_path_csv = "C:/Users/loren/Documents/Data/eegData/eeg_data.csv"

    # Computer Marzio
    #path_alcoholismEEG_data = "C://users/Riccardo/Desktop/Marzio/Advanced/Project/"
    #output_path_csv = path_alcoholismEEG_data + "eeg_data.csv"



class utils:
    # generics
    seed = 2024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    num_channels = 64
    train_size = 0.7 
    val_size = 0.2
    test_size = 1 - (train_size + val_size)
    
    # LSTM params
    seq_lenght =  256
    hidden_size = 64
    num_layers = 2
    bidirectional = False
   
    # GCN params
    dim_firstConvGCN = 32
    dim_lastConvGCN = 16

    # CoCoNet params
    batch_size = 8
    num_epochs = 1
    num_classes = 2
