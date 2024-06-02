'''
Reading files to create a single csv
'''

import os
import tarfile
import gzip
import shutil

import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
#sys.path.append("/home/giovanna/Desktop/Lorenzo/...)
from config import user_paths as path


# Funzione per decomprimere i file tar
def decompress_tar_file(tar_path, extract_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

# Funzione per decomprimere i file gz
def decompress_gz_file(gz_path, extract_path):
    output_file_path = extract_path.replace(".gz", "")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return output_file_path



if __name__ == '__main__':
    # Percorso della cartella principale
    base_dir = path.path_alcoholismEEG_data
    
    # Scorro la cartella eeg_full
    for root, _, dirs in os.walk(base_dir):     # in realtà dirs sarebbero i files ma le cartelle compresse sono considerate file
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)

            # Se il nome della cartella finisce con '.tar', decomprimi il file tar
            if ".tar" in dir_name:
                # Estrai il nome del file senza l'estensione .tar e tutto quello che c'è dopo
                base_name = dir_name.split(".tar")[0]
                extract_path = os.path.join(root, base_name)

                decompress_tar_file(dir_path, root)

                #Rimuovi la cartella tar decompressa
                os.remove(dir_path)

                # Scorri attraverso la nuova cartella decompressa
                for sub_root, _, sub_files in os.walk(extract_path):
                    for sub_dir_name in sub_files:
                        sub_dir_path = os.path.join(sub_root, sub_dir_name)

                        # Se il nome della cartella finisce con '.tar', decomprimi il file tar
                        if sub_dir_name.endswith(".gz"):
                            sub_extract_path = decompress_gz_file(sub_dir_path, sub_dir_path)
                            
                            # Rimuovi la cartella tar decompressa
                            os.remove(sub_dir_path)
                            
                            # Scorri attraverso la nuova cartella decompressa e sposta i file fuori
                            for file_root, _, file_files in os.walk(sub_extract_path):
                                for file_name in file_files:
                                    file_path = os.path.join(file_root, file_name)
                                    new_file_path = os.path.join(file_root, file_root.split('/')[-1] + '.' + file_name.split('.')[-1])
                                    os.rename(file_path, new_file_path)
                                    
                                # Rimuovi le cartelle vuote rimaste
                                os.rmdir(sub_extract_path)

    print("Decompressione completata.")





