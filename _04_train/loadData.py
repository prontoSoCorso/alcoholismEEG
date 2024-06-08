import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np

class LoadData(Dataset):
    """
    Custom Dataset class to load and preprocess images from a pandas DataFrame with specific row structure.
    """

    def __init__(self, dataframe, num_channels=64, seq_length=256):
        self.dataframe = dataframe
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.patient_ids = dataframe['Patient'].unique() #[:16] , se voglio prendere solo N pazienti tra i 120 disponibili

    def __len__(self):
        # Calculate the number of graphs based on the number of df_rows and channels
        num_graphs = int(self.dataframe.shape[0] / self.num_channels)
        return num_graphs

    def __getitem__(self, idx):
        """
        Retrieves a graph sample by extracting consecutive rows from the DataFrame.

        Args:
            idx (int): Index of the graph sample (not row index).

        Returns:
            tuple: A tuple containing the preprocessed graph and its label (if available).
        """
        # Calculate the starting row index for the current graph
        start_row = idx * self.num_channels

        # Extract consecutive rows for the current graph
        graph_data = self.dataframe.iloc[start_row:start_row + self.num_channels, 3:-1].values.reshape(-1, self.seq_length)

        # Convert to PyTorch tensor and apply transformations if provided
        graph = torch.from_numpy(graph_data).float()

        # Retrieve label if present (assuming 'label' column)
        label = torch.tensor(self.dataframe.iloc[start_row, -1])

        return graph, label
    


    def split_dataset(self, train_size=0.7, val_size=0.2, test_size=0.1):
        """
        Splits the dataset into training, validation, and test sets.

        Args:
            test_size (float, optional): Proportion of data for the test set (default: 0.1).
            val_size (float, optional): Proportion of data for the validation set (default: 0.2).
            random_seed (int, optional): Random seed for splitting (default: 42).

        Returns:
            tuple: A tuple containing three ImageDataset objects for training, validation, and test sets.
        """

        # Ensure test_size + val_size <= 1
        if test_size + val_size > 1:
            raise ValueError("test_size + val_size must be less than or equal to 1")


        '''
        # Trova i primi 16 pazienti univoci
        unique_patients = self.dataframe['Patient'].unique()[:16]

        # Filtra il DataFrame per includere solo i primi 16 pazienti univoci
        patient_groups = self.dataframe[self.dataframe['Patient'].isin(unique_patients)]

        # Group data by patient ID
        patient_groups = patient_groups.groupby('Patient')
        '''

        # Group data by patient ID
        patient_groups = self.dataframe.groupby('Patient')

        # Create a list to hold data for each split
        data_splits = []
        for _ in range(3):  # Create empty lists for train, val, test
            data_splits.append([])

        # Shuffle patient IDs
        shuffled_ids = list(np.random.permutation(self.patient_ids))

        # Split data by patient into training, validation, and test sets
        for patient_id in shuffled_ids:
            patient_data = patient_groups.get_group(patient_id)

            # Randomly choose which split this patient belongs to (train, val, or test)
            split_choice = torch.multinomial(torch.tensor([train_size, val_size, test_size]), 1)

            data_splits[split_choice.item()].append(patient_data)  # Add patient data to chosen split

        # Create ImageDataset objects for each split
        train_data = pd.concat(data_splits[0])
        val_data = pd.concat(data_splits[1])
        test_data = pd.concat(data_splits[2])

        train_dataset = LoadData(train_data, self.num_channels, self.seq_length)
        val_dataset = LoadData(val_data, self.num_channels, self.seq_length)
        test_dataset = LoadData(test_data, self.num_channels, self.seq_length)

        return train_dataset, val_dataset, test_dataset




