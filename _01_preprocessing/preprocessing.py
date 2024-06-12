import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pywt


import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "alcoholismEEG"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "alcoholismEEG":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)


from config import user_paths as path
from config import utils


"""
Normalization Strategies

    Independent Normalization:
        Normalize the EEG signals of each patient independently.
        This method ensures that the features of each patient's data are scaled relative to their own distribution, which can help in cases where there are significant inter-patient variations.

    Collective Normalization:
        Normalize the EEG signals across all patients together.
        This method scales the data relative to the entire dataset, which can be beneficial when inter-patient variability is not as pronounced or when you want the model to learn a more generalized pattern.

Best Practices and Recommendations

    Independent Normalization:
        Advantages:
            Handles inter-patient variability effectively.
            Ensures that the model does not get biased by the data distribution of specific patients.
        Disadvantages:
            May lose some information about the global distribution of EEG signals across patients.
        When to Use:
            Significant inter-patient variability.
            When individual patient differences are crucial for diagnosis or treatment.

    Collective Normalization:
        Advantages:
            Can leverage global patterns across all patients.
            Useful when the goal is to learn common features applicable to all patients.
        Disadvantages:
            May obscure individual patient characteristics.
            Risk of model bias if certain patients dominate the dataset.
        When to Use:
            When inter-patient variability is minimal or irrelevant.
            For generalized models that aim to identify common features.
"""



# Normalization (z-score)
def normalize_data(data):
    """Normalize the EEG data using z-score normalization."""
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data.T).T  # Transpose to normalize and then transpose back
    return data_normalized



# Denoising
# Denoising could be useful to remove of eliminating artifacts due to muscular movement and blinking as well 
# as swallowing manually.

def denoise_signal(signal, wavelet='db4', level=4, threshold_type='soft', sigma=None):
    """Denoise a signal using Discrete Wavelet Transform (DWT).

    Args:
    signal (numpy.ndarray): Input signal to be denoised.
    wavelet (str): Wavelet type (default: 'db4').
    level (int): Decomposition level (default: 6).
    threshold_type (str): Thresholding type ('soft' or 'hard') (default: 'soft').
    sigma (float): Noise standard deviation (default: None).

    Returns:
    numpy.ndarray: Denoised signal.
    """
    # Decompose signal using DWT
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise standard deviation if not provided
    if sigma is None:
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # Thresholding
    threshold = sigma * np.sqrt(2 * np.log2(len(signal)))
    if threshold_type == 'soft':
        denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    elif threshold_type == 'hard':
        denoised_coeffs = [pywt.threshold(c, threshold, mode='hard') for c in coeffs]

    # Reconstruct denoised signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_signal




def update_patient_data(patient_data):
    # Extract signal data and transpose it to fit the shape (n_channels, n_times)
    signal_timesteps_names = [f'Value_{i}' for i in range(utils.seq_length)]
    data = patient_data[signal_timesteps_names].values

    num_trials = data.shape[0] // utils.num_channels

    # Utilizzare np.hstack e slicing per riorganizzare l'array
    reshaped_data = np.hstack([data[i*utils.num_channels:(i+1)*utils.num_channels] for i in range(num_trials)])
    
    # Normalize the data
    data_normalized = normalize_data(reshaped_data)
    
    # Denoise the normalized data
    denoised_data = denoise_signal(data_normalized)
    
    processed_data = denoised_data.reshape(utils.num_channels, num_trials, utils.seq_length)
    processed_data = processed_data.transpose(1, 0, 2)
    processed_data = processed_data.reshape(utils.num_channels * num_trials, utils.seq_length)


    
    return processed_data


if __name__ == "__main__":
    df = pd.read_csv(path.output_path_csv)

    trials = df["Trial"].unique()

    for trial in trials:
        # Apply the denoising function to each group of patient data and update the original DataFrame
        df_trial = df.loc[df["Trial"] == trial, :]
        df_processed = df_trial.groupby('Patient').apply(update_patient_data)

        unique_patients = df_trial["Patient"].unique()

        for i, patient in enumerate(unique_patients):
            df_trial.loc[df_trial["Patient"] == patient, [f'Value_{j}' for j in range(utils.seq_length)] ] = df_processed[i]


        # Salvare il DataFrame in un file CSV
        output_csv_path = path.output_path_trial_csv + trial + ".csv"
        df_trial.to_csv(output_csv_path, index=False)

    print("===================")


