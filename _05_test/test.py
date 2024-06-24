import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, brier_score_loss, balanced_accuracy_score
from sklearn.utils import resample
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

def calculate_metrics(all_labels, all_predictions, all_probabilities):
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    kappa = cohen_kappa_score(all_labels, all_predictions)
    brier = brier_score_loss(all_labels, all_probabilities)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    return accuracy, precision, recall, f1, kappa, brier, balanced_acc

def bootstrap_metrics(all_labels, all_predictions, all_probabilities, n_bootstraps=1000, alpha=0.95):
    bootstrapped_metrics = []

    for _ in range(n_bootstraps):
        indices = np.random.randint(0, len(all_labels), len(all_labels))
        if len(np.unique(all_labels[indices])) < 2 or len(np.unique(all_predictions[indices])) < 2:
            continue
        
        metrics = calculate_metrics(all_labels[indices], all_predictions[indices], all_probabilities[indices])
        bootstrapped_metrics.append(metrics)

    bootstrapped_metrics = np.array(bootstrapped_metrics)
    lower = np.percentile(bootstrapped_metrics, (1 - alpha) / 2 * 100, axis=0)
    upper = np.percentile(bootstrapped_metrics, (1 + alpha) / 2 * 100, axis=0)
    
    return lower, upper

if __name__ == "__main__":
    
    for selected_file in utils.files:
        if utils.using_GAT:
            best_model_path = "./_05_test/bestGAT_model_" + selected_file + ".pth"
        else:
            best_model_path = "./_05_test/best_model_" + selected_file + ".pth"

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
        test_loader = DataLoader(dataset=test_dataset, batch_size=utils.batch_size, shuffle=False, collate_fn=collate_fn)
        # La dimensione del batch sarà (batch_size, max_depth_in_batch, num_channels, num_features)

        # Definizione del modello, dell'optimizer utilizzato e della Loss Function
        model = CoCoNet.CoCoNet(seq_lenght=utils.seq_length, hidden_size=utils.hidden_size, num_layers=utils.num_layers, bidirectional=utils.bidirectional, dim_lastConvGCN=utils.dim_lastConvGCN, G=G)

        # Carica il modello migliore salvato
        model.load_state_dict(torch.load(best_model_path))
        criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification

        # Stampo modello
        summary(model, input_size=(utils.batch_size, utils.seq_length, utils.num_channels))

        # Valutazione finale sul test set
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            correct_test_predictions = 0
            total_test_samples = 0
            all_labels = []
            all_predictions = []
            all_probabilities = []

            for inputs, labels, num_trials in test_loader:
                inputs = inputs.to(utils.device)
                labels = labels.float().to(utils.device)

                if utils.using_GAT:
                    outputs, _, _ = model(inputs, num_trials)
                else:
                    outputs, _ = model(inputs, num_trials)

                all_labels.append(labels.cpu())
                loss = criterion(torch.squeeze(outputs, 1), labels)
                test_loss += loss.item()

                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                all_predictions.append(predictions.cpu())
                all_probabilities.append(probabilities.cpu())
                correct_test_predictions += (torch.squeeze(predictions, 1) == labels).sum().item()
                total_test_samples += labels.size(0)

        # Calcola loss e accuracy medie sul test set
        test_loss /= len(test_loader)
        test_accuracy = correct_test_predictions / total_test_samples

        all_labels = torch.cat(all_labels).numpy()
        all_predictions = torch.cat(all_predictions).squeeze(1).numpy()
        all_probabilities = torch.cat(all_probabilities).squeeze(1).numpy()

        # Calcola metriche aggiuntive
        accuracy, precision, recall, f1, kappa, brier, balanced_acc = calculate_metrics(all_labels, all_predictions, all_probabilities)

        # Calcola gli intervalli di confidenza
        lower, upper = bootstrap_metrics(all_labels, all_predictions, all_probabilities)

        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy} (95% CI: {lower[0]} - {upper[0]})')
        print(f'Precision: {precision} (95% CI: {lower[1]} - {upper[1]})')
        print(f'Recall: {recall} (95% CI: {lower[2]} - {upper[2]})')
        print(f'F1 Score: {f1} (95% CI: {lower[3]} - {upper[3]})')
        print(f'Cohen\'s Kappa: {kappa} (95% CI: {lower[4]} - {upper[4]})')
        print(f'Brier Score: {brier} (95% CI: {lower[5]} - {upper[5]})')
        print(f'Balanced Accuracy: {balanced_acc} (95% CI: {lower[6]} - {upper[6]})')
