import pandas as pd
import matplotlib.pyplot as plt

# Datei einlesen
file_path = "/Users/anton/CodingProjects/PIPNet_Shortcuts/data/log_epoch_short.csv"  # Pfad zur Datei anpassen
# Spalten manuell definieren, da Datei anscheinend unvollständige numerische Werte hat
columns = [
    "epoch",
    "test_top1_acc",
    "test_top5_acc",
    "almost_sim_nonzeros",
    "local_size_all_classes",
    "almost_nonzeros_pooled",
    "num_nonzero_prototypes",
    "mean_train_acc",
    "mean_train_loss_during_epoch"
]

# Datei einlesen und 'n.a.' als NaN interpretieren
df = pd.read_csv(file_path, sep=",", names=columns, skiprows=1, na_values=["n.a."])

# Epoch als numerischen Wert sichern
#df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

# Plot für jede Spalte erstellen
numeric_columns = df.select_dtypes(include=["float", "int"]).columns
for column in numeric_columns:
    plt.figure(figsize=(20, 10))
    plt.plot(df["epoch"], df[column], marker="o", label=column)
    plt.title(f"Verlauf der Spalte: {column}", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.grid(True)
    #plt.xlim(0,20)  # Setze die x-Achsen-Grenzen
   # plt.ylim(0, 10)
    plt.legend()
    plt.show()
