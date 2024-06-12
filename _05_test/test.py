import pandas as pd
import torch
from torch_geometric.data import DataLoader
import random
import numpy as np

import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "alcoholismEEG"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "alcoholismEEG":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from _04_train import loadData
from _02_graphDefinition import graphCreation
from config import utils
from config import user_paths
from _03_CoCoNetAndLayers import CoCoNet
import plotGraphEmbeddings


if __name__ == "__main__":

    # Importo il CSV e lo converto in un dataframe
    df = pd.read_csv(user_paths.output_path_trial_csv + utils.selected_file + ".csv")

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

    seed_everything(utils.seed)

    # Creo l'oggetto dataset
    dataset = loadData.LoadData(df, utils.num_channels, utils.seq_length)

    # Creo il grafo fisso degli elettrodi
    G = graphCreation.createGraph()

    # Splitto il dataset in training, validation, e test sets
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(utils.train_size, utils.val_size, utils.test_size)

    # Print dimensioni sets
    print(f'Training set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')
    print(f'Test set size: {len(test_dataset)}')

    # Create DataLoader objects for training and validation
    test_loader = DataLoader(test_dataset, batch_size=utils.batch_size, shuffle=False)

    # Definizione del modello, dell'optimizer utilizzato e della Loss Function
    model = CoCoNet.CoCoNet(utils.seq_length, utils.hidden_size, utils.num_layers, utils.bidirectional, utils.dim_lastConvGCN, G)

    # Carica il modello migliore salvato
    model.load_state_dict(torch.load(utils.best_model_path))
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification

    # Valutazione finale sul test set
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0
        gcn_out = []
        all_labels = []

        for inputs, labels in test_loader:
            inputs = inputs.to(utils.device)
            labels = labels.float().to(utils.device)

            outputs, gcn_out_tmp = model(inputs)

            gcn_out.append(gcn_out_tmp.reshape(inputs.size(0), utils.dim_lastConvGCN))
            all_labels.append(labels)

            loss = criterion(torch.squeeze(outputs, 1), labels)
            test_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct_test_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
            total_test_samples += labels.size(0)


    # Stack all the outputs from the GCN
    gcn_out = torch.cat(gcn_out).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy().astype(int)

    # Calcola loss e accuracy medie sul test set
    test_loss /= len(test_loader)
    test_accuracy = correct_test_predictions / total_test_samples

    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    # Plot graphs embeddings (UMAP and t-SNE)
    plotGraphEmbeddings.plot_graph_embeddings(gcn_out, all_labels)