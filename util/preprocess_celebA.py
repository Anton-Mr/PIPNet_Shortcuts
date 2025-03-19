import os
import shutil
import pandas as pd

image_dir = "./data/CelebA/img_align_celeba/img_align_celeba"

df_partition = pd.read_csv("./data/CelebA/list_eval_partition.csv")
df_attribute = pd.read_csv("./data/CelebA/list_attr_celeba.csv")
gender = df_attribute["Male"]
df_partition["Male"] = gender
source_path= "./data/CelebA/img_align_celeba/img_align_celeba"
test_path = "./data/CelebA/test"
train_path = "./data/CelebA/train"
train_test_dict = {0:"train", 1:"test", 2:"test"}
classes_dict = {1:"male", -1:"female_glasses"}
root_dir = "./data/CelebA"
for i,row in df_partition.iterrows():
    source_img_path = os.path.join(source_path, row["image_id"])
    dest_path = os.path.join(root_dir,train_test_dict[row["partition"]],classes_dict[row["Male"]])
    # Zielpfad im Überordner (ohne Unterordnerstruktur)
    shutil.move(source_img_path, dest_path)
