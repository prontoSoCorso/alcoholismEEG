import pandas as pd
import torch
from torch_geometric.data import DataLoader
import random
import numpy as np

import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
import loadData
from _02_graphDefinition import graphCreation
from config import utils
from config import user_paths
from _03_CoCoNetAndLayers import CoCoNet

import networkx as nx


if __name__ == "__main__":

    # Importo il CSV e lo converto in un dataframe
    df = pd.read_csv(user_paths.output_path_csv)

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
    dataset = loadData.LoadData(df, utils.num_channels, utils.seq_lenght)

    # Creo il grafo fisso degli elettrodi
    G = graphCreation.createGraph()

    # Splitto il dataset in training, validation, e test sets
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(utils.train_size, utils.val_size, utils.test_size)

    # Print dimensioni sets
    print(f'Training set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')
    print(f'Test set size: {len(test_dataset)}')


    # Create DataLoader objects for training and validation
    train_loader = DataLoader(train_dataset, batch_size=utils.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=utils.batch_size, shuffle=False)


    # Definizione del modello, dell'optimizer utilizzato e della Loss Function
    model = CoCoNet.CoCoNet(utils.seq_lenght, utils.hidden_size, utils.num_layers, utils.bidirectional, utils.dim_lastConvGCN, G)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()      # (binary cross-entropy for binary classification)

    
    '''
    # start a new wandb run to track this script
    wandb.init(
        # Set the W&B project where this run will be logged
        project=conf.project_name,

        # Track hyperparameters and run metadata
        config={
            "exp_name": conf.exp_name,
            "dataset": conf.dataset,
            "model": conf.model_name,
            "num_epochs": conf.num_epochs,
            "batch_size": conf.batch_size,
            "learning_rate": conf.learning_rate,
            "optimizer_type": conf.optimizer_type,
            "img_size": conf.img_size, 
            "num_classes": conf.num_classes
        }
    )
    wandb.run.name = conf.exp_name
    
    '''

    
    for epoch in range(utils.num_epochs):
        model.train()  # Imposta la modalità di training

        epoch_train_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(utils.device)
            labels = labels.float().to(utils.device)

            optimizer.zero_grad()  # Azzeramento dei gradienti, altrimenti ogni volta si aggiungono a quelli calcolati al loop precedente

            # Calcolo output modello
            outputs = model(inputs)

            # Calcola la loss
            loss = criterion(torch.squeeze(outputs,1), labels)

            # Calcola l'accuracy
            predictions = (outputs > 0.5).float()
            correct_train_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
            total_train_samples += labels.size(0)

            # Calcola i gradienti e aggiorna i pesi
            loss.backward()     # The new loss adds to what the last one computed, non lo fa la total loss
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

                outputs = model(inputs)

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

        '''

        wandb.log({'epoch': epoch + 1,
                   'train_accuracy': train_accuracy,
                   'train_loss': train_loss,
                   'val_accuracy': val_accuracy,
                   'val_loss': val_loss})

        '''

        # Stampa delle informazioni sull'epoca
        print(f'Epoch [{epoch+1}/{utils.num_epochs}], Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')





