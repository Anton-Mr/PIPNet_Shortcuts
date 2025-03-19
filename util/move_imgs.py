import os
import shutil

# Der Pfad zum Überordner
base_folder = "/Users/anton/CodingProjects/PIPNet_Shortcuts/data/segmentations"



# Durch alle Unterordner und Dateien iterieren
for root, dirs, files in os.walk(base_folder):
    # Dateien im aktuellen Unterordner
    for file in files:
        # Absoluter Pfad zur Datei
        source_path = os.path.join(root, file)
        # Zielpfad im Überordner (ohne Unterordnerstruktur)
        destination_path = os.path.join(base_folder, file)

        # Datei verschieben
        shutil.move(source_path, destination_path)
        print(f"Moved: {source_path} -> {destination_path}")


print("Alle Dateien wurden in den Überordner verschoben.")
