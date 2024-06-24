import pandas as pd
import torch
from torch.utils.data import DataLoader

import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "alcoholismEEG"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "alcoholismEEG":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from _02_graphDefinition import graphCreation, attentionGraph
from _03_CoCoNetAndLayers import CoCoNet
from _04_train import loadData
from config import utils
from config import user_paths

import plotUMAPandTSNE, plotUMAP, gridSearchUMAP

from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV


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

        # Create DataLoader objects for training and validation
        loader = DataLoader(dataset=dataset, batch_size=utils.batch_size, shuffle=False, collate_fn=collate_fn)
        # La dimensione del batch sarà (batch_size, max_depth_in_batch, num_channels, num_features)

        # Definizione del modello, dell'optimizer utilizzato e della Loss Function
        model = CoCoNet.CoCoNet(utils.seq_length, utils.hidden_size, utils.num_layers, utils.bidirectional, utils.dim_lastConvGCN, G)

        # Carica il modello migliore salvato
        model.load_state_dict(torch.load(best_model_path))
        criterion = torch.nn.BCEWithLogitsLoss()  # Binary cross-entropy for binary classification

        # Valutazione finale sul test set
        model.eval()
        with torch.no_grad():
            gcn_out = []
            all_labels = []
            if utils.using_GAT:
                all_weights = []
                all_num_trials = []

            for inputs, labels, num_trials in loader:
                inputs = inputs.to(utils.device)
                labels = labels.float().to(utils.device)

                if utils.using_GAT:
                    outputs, gcn_out_tmp, (attention_index, attention_weights) = model(inputs, num_trials)
                    all_weights.append(attention_weights)
                    all_num_trials.append(sum(num_trials))
                else:
                    outputs, gcn_out_tmp = model(inputs, num_trials)

                gcn_out.append(gcn_out_tmp)
                all_labels.append(labels)

                loss = criterion(torch.squeeze(outputs, 1), labels)
                predictions = (outputs > 0.5).float()

        
        # Stack all the outputs from the GCN
        gcn_out = torch.cat(gcn_out).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy().astype(int)
        
        '''
        # Plot graphs embeddings (UMAP and t-SNE)
        plotUMAPandTSNE.plot_graph_embeddings(gcn_out, all_labels)
        '''

        # Parameter grid for UMAP
        param_grid = {
            "n_neighbors": [10,20,30,40,50],
            "min_dist": [0.1],
            "n_components": [2],
            "metric": ["euclidean"]
        }

        # Custom scoring function for silhouette score
        def silhouette_scorer(estimator, X, y):
            transformed_X = estimator.fit_transform(X)
            return silhouette_score(transformed_X, y)

        # Grid search
        grid_search = GridSearchCV(gridSearchUMAP.UMAPEstimator(), param_grid, scoring=silhouette_scorer, cv=3, n_jobs=1)
        grid_search.fit(gcn_out, all_labels)

        # Best parameters
        print("Best parameters found: ", grid_search.best_params_)
        print("Best silhouette score: ", grid_search.best_score_)
        
        # Plot UMAP
        plotUMAP.plot_umap(gcn_out, all_labels, grid_search.best_params_)

        if utils.using_GAT:
            num_attention_weights = len(G.edges)+utils.num_channels

            num_edges = len(G.edges)

            self_attention_weights = []
            for i, weights in enumerate(all_weights):
                self_attention_weights.append(weights[(all_num_trials[i]*num_edges):,0].unsqueeze(1))
                all_weights[i] = weights[:(all_num_trials[i]*num_edges),0].unsqueeze(1)


            attention_weights = attentionGraph.average_subtensors(all_weights, num_edges).squeeze(1).tolist()
            self_attention_weights = attentionGraph.average_subtensors(self_attention_weights, utils.num_channels).squeeze(1).tolist()
            
            attentionGraph.plot_weight_difference(G, attention_weights)
