import os
import random
import shutil

DATASET_FOLDER = 'dataBalenced'
SORTED_FOLDER = 'dataSorted'
MAX_PER_CLASS = 20

os.makedirs(DATASET_FOLDER, exist_ok=True)

if os.path.exists('dataBalenced'):
    shutil.rmtree('dataBalenced')

for label in os.listdir(SORTED_FOLDER):
    class_folder = os.path.join(SORTED_FOLDER, label)
    if not os.path.isdir(class_folder):
        continue

    files = os.listdir(class_folder) #get files
    random.shuffle(files)

    files = files[:MAX_PER_CLASS] #limit amt of files

        #|| sends sorted files to new loc
        #\/
    target_folder = os.path.join(DATASET_FOLDER, label)
    os.makedirs(target_folder, exist_ok=True)
    for f in files:
        src = os.path.join(class_folder, f)
        dst = os.path.join(target_folder, f)
        shutil.copy(src, dst)

print('Balanced dataset saved in:', DATASET_FOLDER)
