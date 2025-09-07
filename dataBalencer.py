import os
import random
import shutil

DATASET_FOLDER = 'dataBalenced'
SORTED_FOLDER = 'dataSorted'
MAX_PER_CLASS = 25
TEST_PERCENT = 0.2


if os.path.exists('dataBalenced'):
    shutil.rmtree('dataBalenced')
os.makedirs(DATASET_FOLDER, exist_ok=True)

for label in os.listdir(SORTED_FOLDER):
    class_folder = os.path.join(SORTED_FOLDER, label)
    if not os.path.isdir(class_folder):
        continue
    if label == "na":
        continue

    files = os.listdir(class_folder) #get files
    random.shuffle(files)

    files = files[:MAX_PER_CLASS] #limit amt of files
    if(label == "board"):
        files = files[:MAX_PER_CLASS*2]
    
    splitIdx = int(len(files)*(1-TEST_PERCENT))
    trainFiles = files[:splitIdx]
    testFiles = files[splitIdx:]

        #|| sends sorted files to new loc
        #\/
    target_folder_train = os.path.join(DATASET_FOLDER, "train", label)
    os.makedirs(target_folder_train, exist_ok=True)
    for f in trainFiles:
        src = os.path.join(class_folder, f)
        dst = os.path.join(target_folder_train, f)
        shutil.copy(src, dst)
    target_folder_test = os.path.join(DATASET_FOLDER, "test", label)
    os.makedirs(target_folder_test, exist_ok=True)
    for f in testFiles:
        src = os.path.join(class_folder, f)
        dst = os.path.join(target_folder_test, f)
        shutil.copy(src, dst)
    

print('Balanced dataset saved in:', DATASET_FOLDER)
