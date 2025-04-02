#%%
import pandas as pd
import numpy as np
import re
import os


# %%


# Merge

def merge_csv(path):

    # Lista dei file CSV nella directory
    file_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

    # Variabile per memorizzare il DataFrame unito
    merged_df = None

    # Loop attraverso i file
    for file in file_paths:
        # Lettura del CSV
        df = pd.read_csv(file)

        # Ottieni il nome del file senza estensione e suffisso dopo il primo trattino
        file_id = os.path.basename(file).split('-')[0]

        # Rinomina le colonne diverse da 'Time'
        new_cols = {col: f"{col}_{file_id}" for col in df.columns if col != 'Time'}
        df.rename(columns=new_cols, inplace=True)

        # Unisci i DataFrame sulla colonna 'Time'
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='Time', how='outer')

    # Salvataggio del file CSV finale
    return merged_df

# Labels

def generate_labels(df, value):
    classification_values = [value]* len(df)  # Cambia qui con la classificazione desiderata (es. 1 o 0)
    labels = df[['Time']].copy()  # Copia solo la colonna 'Time'
    labels['ClassificationValue'] = classification_values
    return labels


# Clean and Convert
def convert_and_clean(value):
    if isinstance(value, str):
        value = value.lower().strip()
        
        # Memory: Conversion of KB, MB, GB in MB
        mem_match = re.match(r"([\d\.,]+)\s*(b|kb|kib|mb|mib|gb|gib)?$", value)
        if mem_match:
            num, unit = float(mem_match.group(1).replace(',', '.')), mem_match.group(2).lower()
            conversion = {
                "b": 1 / 1024 / 1024,
                "kb": 1 / 1024, "kib": 1 / 1024,
                "mb": 1, "mib": 1,
                "gb": 1024, "gib": 1024
            }
            return num * conversion[unit]  # Restituisce MB
        
        # Tempo: Conversion of ms, µs, ns in s
        time_match = re.match(r"([\d\.,]+)\s*(µs|us|ms|s)?$", value)
        if time_match:
            num, unit = float(time_match.group(1).replace(',', '.')), time_match.group(2).lower()
            conversion = {"µs": 1e-6, "us": 1e-6, "ms": 1 / 1000, "s": 1}
            return num * conversion[unit]  # Restituisce secondi

        cleaned = re.sub(r'[%a-zA-Zµ/]+', '', value).strip()
        return cleaned if cleaned else None
    return value


# %%

file_path = ""
output_path = ""
df = merge_csv(file_path);
df.to_csv(output_path, index=False)


# %%
value = 1
output_path_labels = ""
labels = generate_labels(df, value)
labels.to_csv(output_path_labels, index=False)



# %%

df = pd.read_csv(file_path, low_memory=False)

df.iloc[:, 1:] = df.iloc[:, 1:].applymap(convert_and_clean)

# Handle the ∞
df.iloc[:, 1:] = df.iloc[:, 1:].replace("∞", np.nan)

# Fill the NaN with median
df.iloc[:, 1:] = df.iloc[:, 1:].astype(float) #Casting to float
df.iloc[:, 1:] = df.iloc[:, 1:].fillna(df.iloc[:, 1:].median())

#If unique, drop the column
df = df.loc[:, df.nunique(dropna=True) > 1]

cleaned_path = ""

df.to_csv(cleaned_path, index=False)
