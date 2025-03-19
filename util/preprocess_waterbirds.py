import os
import shutil
from sklearn.model_selection import train_test_split

# Pfad zum Ordner 'waterbirds_OLD' (ersetze dies durch den tatsächlichen Pfad)
base_dir = 'data/waterbird'

# Zielordner für train und test
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Aufteilung Verhältnis (z.B. 80% Training, 20% Test)
train_ratio = 0.8

# Erstelle train und test Verzeichnisse
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Gehe durch jeden Unterordner im Hauptordner (jede Kategorie)
for category in os.listdir(base_dir):
    category_path = os.path.join(base_dir, category)
    if not os.path.isdir(category_path) or category in ['train', 'test']:
        # Überspringt die Ordner 'train' und 'test' selbst und evtl. Dateien im base_dir
        continue

    # Hole alle Bilddateien der Kategorie
    images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

    # Aufteilen in Trainings- und Testsets
    train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)

    # Ordner für die Kategorie in train und test erstellen
    train_category_dir = os.path.join(train_dir, category)
    test_category_dir = os.path.join(test_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(test_category_dir, exist_ok=True)

    # Verschiebe die Trainingsbilder in den entsprechenden train-Kategorie-Ordner
    for image in train_images:
        src_path = os.path.join(category_path, image)
        dst_path = os.path.join(train_category_dir, image)
        shutil.copy2(src_path, dst_path)

    # Verschiebe die Testbilder in den entsprechenden test-Kategorie-Ordner
    for image in test_images:
        src_path = os.path.join(category_path, image)
        dst_path = os.path.join(test_category_dir, image)
        shutil.copy2(src_path, dst_path)

print("Die Bilder wurden erfolgreich in Trainings- und Testordner aufgeteilt.")