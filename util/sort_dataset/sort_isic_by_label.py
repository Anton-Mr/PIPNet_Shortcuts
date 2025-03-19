import csv
import os
import shutil
import pandas as pd
import numpy as np

def organizeImagesByLabel():


    metadata = pd.read_csv("./data/ISIC_ALL/metadata.csv", index_col=0)

    source_test_path = "./data/ISIC_ALL/ISIC/test/benign"

    classes_dict = {0: "benign", 1: "malignant"}
    base_dir = "./data/ISIC_ALL/only_B_with_P/test"
    for root, dirs, files in os.walk(source_test_path):
         # Dateien im aktuellen Unterordner
        for file in files:
            if file.endswith(".jpg"):

                if file.endswith("_2.jpg"):
                    file_metadata = metadata.loc[file.split("_2.jpg")[0]+ ".jpg"]
                else:
                    file_metadata = metadata.loc[file]
                file_class = file_metadata["benign_malignant"]
                file_class_decode = classes_dict[file_class]
                # Datei verschieben
                source_path = os.path.join(source_test_path, file)
                destination_path = os.path.join(base_dir, file_class_decode)
                destination_path = os.path.join(destination_path,file)
                if(file_metadata["patches"] == 1):
                    shutil.copy(source_path, destination_path)
                print(f"Moved: {source_path} -> {destination_path}")


def getImgWithPatches():

    metadata = pd.read_csv("./data/ISIC/metadata.csv", index_col=0)



organizeImagesByLabel()