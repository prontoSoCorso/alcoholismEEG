import time

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



