''' Configuration file for the project'''
import random
import numpy as np
import torch
import math

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
    selected_file = "S1_obj"          # S1_obj, S2_nomatch, S2_match
    best_model_path = "./_05_test/best_model_" + selected_file + ".pth"

    # Funzione per impostare il seed
    def seed_everything(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
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
    kernel_size_1 = 5
    padding_size_1 = math.floor(kernel_size_1/2)
    dim_lastConvGCN = 16
    kernel_size_2 = 3
    padding_size_2 = math.floor(kernel_size_2/2)

    # wandb parameters
    project_name = "CoCoNet_for_alcoholismEEG"
    model_name = 'CoCoNet'
    dataset = "alcoholismEEG"
    batch_size = 4              # Numero di pazienti
    num_classes = 2
    num_epochs = 50
    learning_rate = 0.001
    optimizer_type = "Adam"     # Tipo optimizer utilizzato

    exp_name = dataset + "," + model_name + "," + str(num_epochs) + "," + str(batch_size) + "," + str(learning_rate) + "," + optimizer_type

