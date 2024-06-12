''' Configuration file for the project'''
import torch

class user_paths:
    # Computer Lorenzo
    path_alcoholismEEG_data = "C:/Users/loren/Documents/Data/eegData/eeg_full/"
    output_path_csv = "C:/Users/loren/Documents/Data/eegData/eeg_data.csv"
    output_path_trial_csv = "C:/Users/loren/Documents/Data/eegData/"

    # Computer Marzio
    #path_alcoholismEEG_data = "C://users/Riccardo/Desktop/Marzio/Advanced/Project/"
    #output_path_csv = path_alcoholismEEG_data + "eeg_data.csv"



class utils:
    # generics
    seed = 2024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_file = "S2_match"          # S1_obj, S2_nomatch
    best_model_path = "./_05_test/best_model_" + selected_file + ".pth"

    # DataLoader
    num_channels = 64
    train_size = 0.7 
    val_size = 0.2
    test_size = 1 - (train_size + val_size)
    
    # LSTM params
    seq_length =  256
    hidden_size = 64
    num_layers = 2
    bidirectional = False
   
    # GCN params
    dim_firstConvGCN = 32
    dim_lastConvGCN = 16

    # CoCoNet params
    batch_size = 16
    num_epochs = 3
    num_classes = 2
