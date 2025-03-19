import csv
import os
import shutil
import pandas as pd
import numpy as np

def count_test_benign_patch():


    df_metadata = pd.read_csv("./data/ISIC_ALL/metadata.csv", index_col=0)
    patch = df_metadata["patches"]
    source_test_path = "./data/ISIC_ALL/ISIC/test/benign"

    source_train_path = "./data/ISIC_ALL/ISIC/train/benign"

    source_test_mal_path = "./data/ISIC_ALL/ISIC/test/malignant"

    source_train_mal_path = "./data/ISIC_ALL/ISIC/train/malignant"
    counter = 0

    for root, dirs, files in os.walk(source_test_path):
        for file in files:
            if file!=".DS_Store":
                if patch[file] == 1:
                    counter += 1

    for root, dirs, files in os.walk(source_train_path):
        for file in files:
            if file!=".DS_Store":
                if patch[file] == 1:
                    counter += 1
    for root, dirs, files in os.walk(source_test_mal_path):
        for file in files:
            if file!=".DS_Store":
                if patch[file] == 1:
                    counter += 1

    for root, dirs, files in os.walk(source_train_mal_path):
        for file in files:
            if file!=".DS_Store":
                if patch[file] == 1:
                    counter += 1


    return counter

count = count_test_benign_patch()
print(count)