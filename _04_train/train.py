import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb

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

    for selected_file in utils.files:
        best_model_path = "./_05_test/bestGAT_model_" + selected_file + ".pth"
        exp_name = utils.exp_name + "," + selected_file

        # Importo il CSV e lo converto in un dataframe
        df = pd.read_csv(user_paths.output_path_trial_csv + selected_file + ".csv")

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
        train_loader = DataLoader(dataset=train_dataset, batch_size=utils.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(dataset=val_dataset, batch_size=utils.batch_size, shuffle=False, collate_fn=collate_fn)
        # La dimensione del batch sarà (batch_size, max_depth_in_batch, num_channels, num_features)

        # Definizione del modello, dell'optimizer utilizzato e della Loss Function
        model = CoCoNet.CoCoNet(utils.seq_length, utils.hidden_size, utils.num_layers, utils.bidirectional, utils.dim_lastConvGCN, G)
        optimizer = torch.optim.Adam(model.parameters(), lr=utils.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification

        # Path to save the best model checkpoint
        best_val_loss = float('inf')

        # start a new wandb run to track this script
        wandb.init(
            # Set the W&B project where this run will be logged
            project=utils.project_name,

            # Track hyperparameters and run metadata
            config={
                "exp_name": exp_name,
                "dataset": utils.dataset,
                "model": utils.model_name,
                "trialType": selected_file,
                "num_epochs": utils.num_epochs,
                "batch_size": utils.batch_size,
                "learning_rate": utils.learning_rate,
                "optimizer_type": utils.optimizer_type,
                "num_classes": utils.num_classes
            }
        )
        wandb.run.name = exp_name

        for epoch in range(utils.num_epochs):
            model.train()  # Imposta la modalità di training

            epoch_train_loss = 0.0
            correct_train_predictions = 0
            total_train_samples = 0

            for inputs, labels, num_trials in train_loader:
                inputs = inputs.to(utils.device)
                labels = labels.float().to(utils.device)

                optimizer.zero_grad()  # Azzeramento dei gradienti

                # Calcolo output modello
                if utils.using_GAT:
                    outputs, _, _ = model(inputs, num_trials)
                else:
                    outputs, _ = model(inputs, num_trials)

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

                for inputs, labels, num_trials in val_loader:
                    inputs = inputs.to(utils.device)
                    labels = labels.float().to(utils.device)

                    if utils.using_GAT:
                        outputs, _, _ = model(inputs, num_trials)
                    else:
                        outputs, _ = model(inputs, num_trials)

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
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved at epoch {epoch + 1}")

            wandb.log({'epoch': epoch + 1,
                    'train_accuracy': train_accuracy,
                    'train_loss': train_loss,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss})

            # Stampa delle informazioni sull'epoca
            print(f'Epoch [{epoch+1}/{utils.num_epochs}], Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')


        # Closing Wandb session for each for
        wandb.finish()
