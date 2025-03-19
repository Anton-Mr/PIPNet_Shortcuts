import csv
import os
import shutil
import pandas as pd
import numpy as np

def organizeImagesByPatch():


    df_metadata = pd.read_csv("./data/ISIC_ALL/metadata.csv", index_col=0)
    patch = df_metadata["patches"]
    source_test_path = "./data/ISIC_ALL/ISIC/test/malignant"

    dest_root_dir = "./data/ISIC_ALL/ISIC_4G/test/"
    for root, dirs, files in os.walk(source_test_path):
        for file in files:

            source_path = os.path.join(source_test_path, file)
            if patch[file] == 0:
                dest_dir = os.path.join(dest_root_dir,"malignant_np")
            else :
                dest_dir = os.path.join(dest_root_dir,"malignant_p")

            shutil.copy(source_path, dest_dir)


def getImageDetail():

    df_metadata = pd.read_csv("./data/ISIC_ALL/metadata.csv", index_col=0)
    df_metadata["split"] = np.repeat(3,len(df_metadata))




#organizeImagesByPatch()