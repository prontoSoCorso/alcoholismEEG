import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "alcoholismEEG"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "alcoholismEEG":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from _02_graphDefinition import graphCreation
from _03_CoCoNetAndLayers import CoCoNet
from _04_train import loadData
from config import utils
from config import user_paths


def collate_fn(batch):
    # Creo vettore riassuntivo della profondità di ogni paziente
    num_trials = [x[0].size(0) for x in batch]

    # Trova la massima profondità nel batch
    max_depth = max(num_trials)
    num_channels = batch[0][0].size(1)
    seq_length = batch[0][0].size(2)

    # Crea un batch con padding
    padded_batch = torch.zeros((len(batch), max_depth, num_channels, seq_length))
    labels = torch.zeros(len(batch))

    for i, (tensor, label) in enumerate(batch):
        depth = tensor.size(0)
        padded_batch[i, :depth, :, :] = tensor
        labels[i] = label

    return padded_batch, labels, num_trials



if __name__ == "__main__":

    # Importo il CSV e lo converto in un dataframe
    df = pd.read_csv(user_paths.output_path_trial_csv + utils.single_file + ".csv")

    # Funzione per impostare il seed
    utils.seed_everything(utils.seed)

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
    test_loader = DataLoader(dataset=test_dataset, batch_size=utils.batch_size, shuffle=False, collate_fn=collate_fn)
    # La dimensione del batch sarà (batch_size, max_depth_in_batch, num_channels, num_features)

    # Definizione del modello, dell'optimizer utilizzato e della Loss Function
    model = CoCoNet.CoCoNet(utils.seq_length, utils.hidden_size, utils.num_layers, utils.bidirectional, utils.dim_lastConvGCN, G)

    # Carica il modello migliore salvato
    model.load_state_dict(torch.load(utils.best_model_path))
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification

    # Stampo modello
    summary(model, input_size=(utils.batch_size, utils.seq_length, utils.num_channels))

    # Valutazione finale sul test set
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct_test_predictions = 0
        total_test_samples = 0
        gcn_out = []
        all_labels = []

        for inputs, labels, num_trials in test_loader:
            inputs = inputs.to(utils.device)
            labels = labels.float().to(utils.device)

            outputs, gcn_out_tmp = model(inputs, num_trials)

            gcn_out.append(gcn_out_tmp.reshape(inputs.size(0), utils.dim_lastConvGCN))
            all_labels.append(labels)

            loss = criterion(torch.squeeze(outputs, 1), labels)
            test_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct_test_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
            total_test_samples += labels.size(0)

    # Calcola loss e accuracy medie sul test set
    test_loss /= len(test_loader)
    test_accuracy = correct_test_predictions / total_test_samples

    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    '''
    # Accedi ai pesi dello strato GATConv
    gat_conv_weights = model.GCN.conv2
    print(gat_conv_weights)
    
    # Crea il grafo con i pesi aggiornati
    G_with_weights = graphCreation.createGraphWithWeights(gat_conv_weights)
    
    # Plotta il grafo con i pesi aggiornati
    graphCreation.plotGraph(G_with_weights)
    '''