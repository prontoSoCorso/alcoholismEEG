import pandas as pd
import torch
from torch.utils.data import DataLoader
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

import loadData
from _02_graphDefinition import graphCreation
from config import utils
from config import user_paths
from _03_CoCoNetAndLayers import CoCoNet


def collate_fn(batch):
    # Trova la massima profondità nel batch
    max_depth = max([x[0].size(0) for x in batch])
    num_channels = batch[0][0].size(1)
    seq_length = batch[0][0].size(2)

    # Crea un batch con padding
    padded_batch = torch.zeros((len(batch), max_depth, num_channels, seq_length))
    labels = torch.zeros(len(batch))

    for i, (tensor, label) in enumerate(batch):
        depth = tensor.size(0)
        padded_batch[i, :depth, :, :] = tensor
        labels[i] = label

    return padded_batch, labels


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
    train_loader = DataLoader(dataset=train_dataset, batch_size=utils.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=utils.batch_size, shuffle=False, collate_fn=collate_fn)
    # La dimensione del batch sarà (batch_size, max_depth_in_batch, num_channels, num_features)

    # Definizione del modello, dell'optimizer utilizzato e della Loss Function
    model = CoCoNet.CoCoNet(utils.seq_length, utils.hidden_size, utils.num_layers, utils.bidirectional, utils.dim_lastConvGCN, G)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification

    # Path to save the best model checkpoint
    best_val_loss = float('inf')

    for epoch in range(utils.num_epochs):
        model.train()  # Imposta la modalità di training

        epoch_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(utils.device)
            labels = labels.float().to(utils.device)

            optimizer.zero_grad()  # Azzeramento dei gradienti

            # Calcolo output modello
            outputs, _ = model(inputs)

            # Calcola la loss
            loss = criterion(torch.squeeze(outputs, 1), labels)

            # Calcola l'accuracy
            predictions = (outputs > 0.5).float()
            correct_train_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
            total_train_samples += labels.size(0)

            # Calcola i gradienti e aggiorna i pesi
            loss.backward()
            epoch_train_loss += loss.item()
            optimizer.step()

        # Valutazione della rete su validation
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            correct_val_predictions = 0
            total_val_samples = 0

            for inputs, labels in val_loader:
                inputs = inputs.to(utils.device)
                labels = labels.float().to(utils.device)

                outputs, _ = model(inputs)

                loss = criterion(torch.squeeze(outputs, 1), labels)
                epoch_val_loss += loss.item()

                predictions = (outputs > 0.5).float()
                correct_val_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
                total_val_samples += labels.size(0)

        # Calcola loss e accuracy per epoca sul train e validation set
        epoch_val_loss /= len(val_loader)
        epoch_train_loss /= len(train_loader)

        val_loss = epoch_val_loss
        val_accuracy = correct_val_predictions / total_val_samples

        train_loss = epoch_train_loss
        train_accuracy = correct_train_predictions / total_train_samples

        # Salva il modello migliore basato sulla validazione
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), utils.best_model_path)
            print(f"Model saved at epoch {epoch + 1}")

        # Stampa delle informazioni sull'epoca
        print(f'Epoch [{epoch+1}/{utils.num_epochs}], Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

