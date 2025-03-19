import csv
import os
import shutil
import pandas as pd
import numpy as np

def organizeImagesByGenderAndGlasses():


    df_partition = pd.read_csv("./data/CelebA/list_eval_partition.csv")
    df_attribute = pd.read_csv("./data/CelebA/list_attr_celeba.csv")
    gender = df_attribute["Male"]
    glasses = df_attribute["Eyeglasses"]
    df_partition["Male"] = gender
    df_partition["Eyeglasses"] = glasses
    source_path = "/Users/anton/Downloads/archive_2/img_align_celeba/img_align_celeba"
    root_dir = "./data/CelebA_nach_glasses"
    train_test_dict = {0: "train", 1: "test", 2: "test"}
    classes_dict = {1: "male", -1: "female"}
    glasses_dict = {1: "glasses", -1:"no_glasses"}

    for i, row in df_partition.iterrows():
        source_img_path = os.path.join(source_path, row["image_id"])
        dest_path = os.path.join(root_dir, train_test_dict[row["partition"]], classes_dict[row["Male"]]+"_"+glasses_dict[row["Eyeglasses"]])
        # Zielpfad im Überordner (ohne Unterordnerstruktur)
        shutil.copy(source_img_path, dest_path)


organizeImagesByGenderAndGlasses()