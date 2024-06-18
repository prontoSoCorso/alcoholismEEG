import torch
from torch.utils.data import Dataset
import numpy as np

class LoadData(Dataset):
    """
    Custom Dataset class to load and preprocess data from a pandas DataFrame with specific row structure.
    """

    def __init__(self, dataframe, num_channels=64, seq_length=256):
        self.dataframe = dataframe
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.patient_ids = dataframe['Patient'].unique() # Prendi solo N pazienti se necessario, ad es. [:16]

        # Raggruppa i dati per paziente
        self.patient_data = []
        for patient_id in self.patient_ids:
            patient_df = dataframe[dataframe['Patient'] == patient_id]

            patient_graphs = []
            label = torch.tensor(patient_df.iloc[0, -1]).float()
            num_graphs = int(patient_df.shape[0] / num_channels)
            for idx in range(num_graphs):
                start_row = idx * num_channels
                graph_data = patient_df.iloc[start_row:start_row + num_channels, 3:-1].values.reshape(num_channels, seq_length)
                graph = torch.from_numpy(graph_data).float()
                patient_graphs.append(graph)
            self.patient_data.append((torch.stack(patient_graphs), label))

    def __len__(self):
        # Numero di pazienti
        return len(self.patient_data)

    def __getitem__(self, idx):
        """
        Retrieves the data and labels for a specific patient.

        Args:
            idx (int): Index of the patient.

        Returns:
            tuple: A tuple containing the preprocessed graphs and their labels.
        """
        patient_graphs, labels = self.patient_data[idx]
        return patient_graphs, labels
    


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
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size, val_size, and test_size must sum to 1")


        '''
        # Trova i primi 16 pazienti univoci
        unique_patients = self.dataframe['Patient'].unique()[:16]

        # Filtra il DataFrame per includere solo i primi 16 pazienti univoci
        patient_groups = self.dataframe[self.dataframe['Patient'].isin(unique_patients)]

        # Group data by patient ID
        patient_groups = patient_groups.groupby('Patient')
        '''

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
        '''

        # Shuffle patient IDs
        shuffled_ids = np.random.permutation(self.patient_ids)

        # Split data by patient into training, validation, and test sets
        num_train = int(train_size * len(shuffled_ids))
        num_val = int(val_size * len(shuffled_ids))

        train_ids = shuffled_ids[:num_train]
        val_ids = shuffled_ids[num_train:num_train + num_val]
        test_ids = shuffled_ids[num_train + num_val:]

        train_data = self.dataframe[self.dataframe['Patient'].isin(train_ids)]
        val_data = self.dataframe[self.dataframe['Patient'].isin(val_ids)]
        test_data = self.dataframe[self.dataframe['Patient'].isin(test_ids)]

        train_dataset = LoadData(train_data, self.num_channels, self.seq_length)
        val_dataset = LoadData(val_data, self.num_channels, self.seq_length)
        test_dataset = LoadData(test_data, self.num_channels, self.seq_length)

        return train_dataset, val_dataset, test_dataset




