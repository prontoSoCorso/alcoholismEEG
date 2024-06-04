import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import sys
sys.path.append("C:/Users/loren/OneDrive - Università di Pavia/Magistrale - Sanità Digitale/alcoholismEEG/")
#sys.path.append("/home/giovanna/Desktop/Lorenzo/...)
from config import user_paths

df = pd.read_csv(user_paths.output_path_csv)

print(df.columns)  # Stampa i nomi delle colonne

# Imposta lo stile di Seaborn
sns.set(style='whitegrid')

# Crea un grafico a barre per visualizzare il numero di segnali per ogni paziente
plt.figure(figsize=(9, 5))  # Imposta la dimensione della figura
grid = sns.histplot(df.groupby('Patient')['Trial'].count()/64, kde=False, color='c')  # Crea il grafico

# Imposta i titoli e le etichette degli assi
plt.title('Number of Signals for Each Patient')
plt.xlabel('Number of Signals')
plt.ylabel('Count of Patients')

# Mostra il grafico
plt.show()

