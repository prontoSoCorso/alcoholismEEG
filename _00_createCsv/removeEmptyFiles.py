import os

import sys
import time
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
#sys.path.append("/home/giovanna/Desktop/Lorenzo/...)
from config import user_paths as path


def remove_files_with_few_lines(directory, min_lines=5):
    # Scorri tutti i file nella directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Verifica se è un file
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                # Se il file ha meno di min_lines righe, eliminalo
                if len(lines) < min_lines:
                    # Chiudi il file e aspetta un attimo prima di eliminarlo
                    file.close()
                    time.sleep(1)
                    os.remove(file_path)
                    print(f"File rimosso: {file_path}")


if __name__ == "__main__":
    base_dir = path.path_alcoholismEEG_data

    for patient_folder in os.listdir(base_dir):
        patient_path = os.path.join(base_dir, patient_folder)

        if os.path.isdir(patient_path):  # Verifica se patient_path è una directory
            remove_files_with_few_lines(patient_path)



