import pandas as pd

import os
# Ottieni il percorso del file corrente
current_file_path = os.path.abspath(__file__)
# Risali la gerarchia fino alla cartella "alcoholismEEG"
parent_dir = os.path.dirname(current_file_path)
while not os.path.basename(parent_dir) == "alcoholismEEG":
    parent_dir = os.path.dirname(parent_dir)
import sys
sys.path.append(parent_dir)

from config import user_paths

# Importo il CSV e lo converto in un dataframe
df = pd.read_csv(user_paths.output_path_csv)

print(df.columns)  # print columns name
print(df['Channel'].unique())

my_channels = (['FP1', 'FP2', 'F7', 'F8', 'AF1', 'AF2', 'FZ', 'F4', 'F3', 'FC6',
       'FC5', 'FC2', 'FC1', 'T8', 'T7', 'CZ', 'C3', 'C4', 'CP5', 'CP6',
       'CP1', 'CP2', 'P3', 'P4', 'PZ', 'P8', 'P7', 'PO2', 'PO1', 'O2',
       'O1', 'X', 'AF7', 'AF8', 'F5', 'F6', 'FT7', 'FT8', 'FPZ', 'FC4',
       'FC3', 'C6', 'C5', 'F2', 'F1', 'TP8', 'TP7', 'AFZ', 'CP3', 'CP4',
       'P5', 'P6', 'C1', 'C2', 'PO7', 'PO8', 'FCZ', 'POZ', 'OZ', 'P2',
       'P1', 'CPZ', 'nd', 'Y'])

my_channels2 = ['FP1', 'FPZ', 'FP2', 
               'AF7', 'AF1', 'AFZ', 'AF2', 'AF8',
               'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
               'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
               'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
               'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 
               'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
               'PO7', 'PO1', 'POZ', 'PO2', 'PO8',
               'O2', 'OZ', 'O1',

               'X', 'nd', 'Y'
               ]

# We don't have P9, P10, IZ
# Probably X is P9, Y is P10 and Iz is nd in the classic EEG configuration with 64 electrods

their_channels = (["FP1","FPZ","FP2",
                   "AF7","AF3","AFZ","AF4","AF8",
                   "F7","F5","F3","F1","FZ","F2","F4","F6","F8",
                   "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
                   "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
                   "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
                   "P9","P7","P5","P3","P1","PZ","P2","P4","P6","P8","P10",
                   "PO7","PO3","POZ","PO4","PO8",
                   "O1","OZ","O2",
                   "IZ"])
