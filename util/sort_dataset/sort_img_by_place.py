import csv
import os
import shutil
import pandas as pd
import numpy as np



def organizeImagesByPlace(image_dir, project_dir):
    df = pd.read_csv("./util/metadata.csv")

    # Erstellen der Ausgabeverzeichnisse
    #train_dir = os.path.join(project_dir, "train")
    #test_dir = os.path.join(project_dir, "test")
    #os.makedirs(train_dir, exist_ok=True)
    #os.makedirs(test_dir, exist_ok=True)
    # Iteration über die Datenzeilen

    train_lb_imgs = df[(df["split"] != 2) & (df["y"] == 0) ].img_filename
    test_lb_imgs = df[(df["split"] == 2) & (df["y"] == 0)].img_filename
    train_wb_imgs = df[(df["split"] != 2) & (df["y"] == 1)].img_filename
    test_wb_imgs = df[(df["split"] == 2) & (df["y"] == 1)].img_filename

    for img in train_lb_imgs:
        shutil.copy(os.path.join(image_dir + "/" + img.split('/')[1]),
                    os.path.join(project_dir + "train/landbird/" + img.split('/')[1]))
    for img in test_lb_imgs:
        shutil.copy(os.path.join(image_dir + "/" + img.split('/')[1]),
                    os.path.join(project_dir + "test/landbird/" + img.split('/')[1]))
    for img in train_wb_imgs:
        shutil.copy(os.path.join(image_dir + "/" + img.split('/')[1]),
                    os.path.join(project_dir + "train/waterbird/" + img.split('/')[1]))
    for img in test_wb_imgs:
        shutil.copy(os.path.join(image_dir + "/" + img.split('/')[1]),
                    os.path.join(project_dir + "test/waterbird/" + img.split('/')[1]))



#organizeImagesByPlace("./data/WATERBIRD_ALL/waterbird_park", "./data/WATERBIRD_ALL/waterbird/")

def organizeImagesByPlaceAndBackground(csv_path, image_dir, project_dir):
    """
    Organisiert Bilder in Ordner 'waterbird' und 'waterbird' basierend auf dem 'place'-Wert in der CSV.

    Args:
        csv_path (str): Pfad zur CSV-Datei.
        image_dir (str): Verzeichnis, in dem sich die Bilder befinden.
        output_dir (str): Basis-Ausgabeverzeichnis für 'waterbird' und 'waterbird'.
    """

    # CSV einlesen
    df = pd.read_csv(csv_path)

    # Erstellen der Ausgabeverzeichnisse
    train_dir = os.path.join(project_dir, "train")
    test_dir = os.path.join(project_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    # Iteration über die Datenzeilen


    train_lb_water_imgs = df[(df["split"]!= 2) & (df["y"]==0) & (df["place"] == 1)].img_filename
    train_lb_land_imgs = df[(df["split"]!= 2 ) & (df["y"]==0 )&  (df["place"] == 0)].img_filename

    test_lb_water_imgs = df[(df["split"]== 2) & (df["y"]==0) & (df["place"] == 1)].img_filename
    test_lb_land_imgs = df[(df["split"]== 2) & (df["y"]==0) & (df["place"] == 0)].img_filename

    train_wb_water_imgs = df[(df["split"] != 2) & (df["y"] == 1)  & (df["place"] == 1)].img_filename
    train_wb_land_imgs = df[(df["split"] != 2) & (df["y"] == 1) & (df["place"] == 0)].img_filename

    test_wb_water_imgs = df[(df["split"] == 2) & (df["y"] == 1 )& (df["place"] == 1)].img_filename
    test_wb_land_imgs = df[(df["split"] == 2) & (df["y"] == 1) & (df["place"] == 0)].img_filename

    '''for img in train_lb_water_imgs:
        shutil.copy(os.path.join(image_dir+"/"+img),os.path.join(project_dir+"train/landbird_water/"+img.split('/')[1]))
    for img in train_lb_land_imgs:
        shutil.copy(os.path.join(image_dir +"/"+ img), os.path.join(project_dir + "train/landbird_land/" + img.split('/')[1]))'''
    for img in test_lb_water_imgs:
        shutil.copy(os.path.join(image_dir+"/"+img.split('/')[1]),os.path.join(project_dir+"test/landbird_water/"+img.split('/')[1]))
    for img in test_lb_land_imgs:
        shutil.copy(os.path.join(image_dir+"/"+img.split('/')[1]),os.path.join(project_dir+"test/landbird_land/"+img.split('/')[1]))
    '''for img in train_wb_water_imgs:
        shutil.copy(os.path.join(image_dir + "/"+img),os.path.join(project_dir + "train/waterbird_water/" + img.split('/')[1]))
    for img in train_wb_land_imgs:
        shutil.copy(os.path.join(image_dir + "/"+img), os.path.join(project_dir + "train/waterbird_land/" + img.split('/')[1]))'''
    for img in test_wb_water_imgs:
        shutil.copy(os.path.join(image_dir + "/"+img.split('/')[1]), os.path.join(project_dir + "test/waterbird_water/" + img.split('/')[1]))
    for img in test_wb_land_imgs:
        shutil.copy(os.path.join(image_dir + "/"+img.split('/')[1]), os.path.join(project_dir + "test/waterbird_land/" + img.split('/')[1]))
# Beispielaufruf
#csv_path = "./util/metadata.csv"  # Ersetze mit dem tatsächlichen Pfad zur CSV
#image_dir = "./data/WATERBIRD_ALL/waterbird_park"      # Ersetze mit dem Verzeichnis der Bilder
#output_dir = "./data/WATERBIRD_ALL/waterbirds_4g/"     # Ersetze mit dem gewünschten Ausgabeverzeichnis

#organizeImagesByPlaceAndBackground(csv_path,image_dir,output_dir)

#print(organizeImagesByPlace(csv_path, image_dir, output_dir))
#organizeImagesByPlace("./data/waterbird_alt","./data/waterbird")

def get_acc_per_group(img_pred_dir):
    dict = {}
    file_path = "/home/thielant/PIPNet/util/metadata.csv"

    """
    Liest eine CSV-Datei Zeile für Zeile ein und speichert die Zeilen in einer Liste.

    Args:
        file_path (str): Pfad zur CSV-Datei.

    Returns:
        list: Liste mit Zeilen (jede Zeile ist eine Liste von Spalten).
    """
    rows_df = pd.read_csv(file_path,delimiter=',', index_col=0, header=0)
    anz_per_group = np.array([[[0],[0]],
                     [[0],[0]]])
    score_per_group = np.array([[[0],[0]],
                     [[0],[0]]])
    for row in rows_df.itertuples():
        img_filename = row.img_filename.split("/", 1)[1]
        img_label = row.y #0 for ?? 1 for ??
        img_background = row.place #0 for ?? 1 for ??

        anz_per_group[img_label][img_background] += 1
        dict[img_filename] = {"label" :row.y, "place": row.place }
    for key, label in img_pred_dir.items():
        entry = dict[key]
        if(entry["label"] ==label):
            score_per_group[label][entry["place"]] += 1

    result = score_per_group /anz_per_group

    return result[0][0],result[0][1],result[1][0],result[1][1]
# erster index ist das label also 1=wb oder 0=lb und zweiter index ist background 1 = waterbird und 0 = waterbird
