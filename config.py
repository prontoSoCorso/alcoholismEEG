''' Configuration file for the project'''
import random
import numpy as np
import torch
import networkx as nx
import sklearn
import umap
import matplotlib.pyplot as plt
import wandb
import math

import wandb.util

class user_paths:
    # Computer Lorenzo
    path_alcoholismEEG_data = "C:/Users/loren/Documents/Data/eegData/eeg_full/"
    output_path_csv = "C:/Users/loren/Documents/Data/eegData/eeg_data.csv"
    #output_path_trial_csv = "C:/Users/loren/Documents/Data/eegData/"
    output_path_trial_csv = "/home/giovanna/Documents/Data/eegData/"

    # Computer Marzio
    #path_alcoholismEEG_data = "C://users/Riccardo/Desktop/Marzio/Advanced/Project/"
    #output_path_csv = path_alcoholismEEG_data + "eeg_data.csv"



class utils:
    # generics
    using_GAT = True

    seed = 2024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = ["S1_obj", "S2_nomatch", "S2_match"]                # "S1_obj", "S2_nomatch", "S2_match"

    def seed_everything(seed=0):
        # Imposta il seed per il generatore di numeri casuali di Python
        random.seed(seed)
        # Imposta il seed per NumPy
        np.random.seed(seed)
        # Imposta il seed per PyTorch
        torch.manual_seed(seed)
        # Se disponibile, imposta il seed per i dispositivi CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Imposta il seed per tutti i dispositivi CUDA
        # Opzioni per PyTorch per garantire la determinazione
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Imposta il seed per NetworkX (indiretto, tramite random)
        random.seed(seed)
        # Imposta il seed per scikit-learn
        sklearn.utils.check_random_state(seed)
        # Imposta il seed per UMAP
        umap.UMAP(random_state=seed)
        # Imposta il seed per matplotlib
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10(np.linspace(0, 1, 10)))

    
    # DataLoader
    num_channels = 64
    train_size = 0.7 
    val_size = 0.1
    test_size = 1 - (train_size + val_size)
    
    # LSTM params
    seq_length =  256
    hidden_size = 64    # Raddoppia quando bidirectional
    num_layers = 2
    bidirectional = False
   
    # GCN params
    dim_firstConvGCN = 32
    kernel_size_1 = 5
    padding_size_1 = math.floor(kernel_size_1/2)
    dim_secondConvGCN = 16
    kernel_size_2 = 3
    padding_size_2 = math.floor(kernel_size_2/2)
    dim_lastConvGCN = 8
    kernel_size_3 = 3
    padding_size_3 = math.floor(kernel_size_3/2)

    # wandb parameters
    project_name = "CoCoNet_for_alcoholismEEG"
    model_name = 'CoCoNet'
    dataset = "alcoholismEEG"
    batch_size = 4              # Numero di pazienti
    num_classes = 2
    num_epochs = 100
    learning_rate = 0.001
    optimizer_type = "Adam"     # Tipo optimizer utilizzato

    exp_name = dataset + "," + model_name + "," + str(num_epochs) + "," + str(batch_size) + "," + str(learning_rate) + "," + optimizer_type

