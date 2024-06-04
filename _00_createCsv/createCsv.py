import os
import pandas as pd

import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
#sys.path.append("/home/giovanna/Desktop/Lorenzo/...)
from config import user_paths as path


# Funzione per processare i file di misurazioni
def process_measurement_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Rimuovi le prime 3 righe commentate
    lines = lines[3:]
    
    # Controllo che i dati siano salvati
    # Estrai il trial ID e la quarta riga commentata
    fourth_line = lines.pop(0).strip()
    trial_info = fourth_line.split(',')[0].split()[1] + '_' + fourth_line.split(',')[0].split()[2]
    
    # Elimina tutte le altre righe commentate
    lines = [line for line in lines if not line.startswith('#')]

    data = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            trial_id, chan, idx, value = parts
            if chan not in data:
                data[chan] = []
            data[chan].append(value)

    return trial_info, data



if __name__ == "__main__":
    # Percorso della cartella principale
    base_dir = path.path_alcoholismEEG_data

    # Lista per memorizzare i dati
    data = []

    # Scorrere i pazienti
    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)
        if os.path.isdir(patient_path):  # Verifica se patient_path è una directory
            # Etichettare il paziente come "trattato" (1) o "controllo" (0)
            label = 1 if patient_folder[3] == 'a' else 0

            # Raccogliere le misurazioni
            for measurement_file in os.listdir(patient_path):
                measurement_path = os.path.join(patient_path, measurement_file)
                if os.path.isfile(measurement_path):  # Verifica se measurement_path è un file
                    trial_info, file_data = process_measurement_file(measurement_path)

                    # Aggiungere i dati alla lista
                    for chan, values in file_data.items():
                        data.append([patient_folder, trial_info, chan] + values + [label])

    # Creare un DataFrame pandas
    columns = ['Patient', 'Trial', 'Channel'] + [f'Value_{i}' for i in range(len(data[0]) - 4)] + ['Label']
    df = pd.DataFrame(data, columns=columns)

    # Salvare il DataFrame in un file CSV
    output_csv_path = path.output_path_csv
    df.to_csv(output_csv_path, index=False)

    print("Dati salvati in formato CSV.")


