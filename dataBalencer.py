import os
import random
import shutil

DATASET_FOLDER = 'dataBalenced'
SORTED_FOLDER = 'dataSorted'
DEFAULT_MAX_PER_CLASS = 40
TEST_PERCENT = 0.2


if os.path.exists('dataBalenced'):
    shutil.rmtree('dataBalenced')
os.makedirs(DATASET_FOLDER, exist_ok=True)

for label in os.listdir(SORTED_FOLDER):
    max_this_class = DEFAULT_MAX_PER_CLASS

    class_folder = os.path.join(SORTED_FOLDER, label)
    if not os.path.isdir(class_folder):
        continue
    if label == "na":
        continue

    files = os.listdir(class_folder) #get files
    random.shuffle(files)

    if(label == "uniqueBoards"):
        label = "board"
    elif(label == "board"):
        max_this_class = 80
    elif(label == "uniqueBishop"):
        label = "bishop"
        max_this_class = 20
    elif(label == "uniqueKings"):
        label = "king"
        max_this_class = 20
    elif(label == "uniqueKnight"):
        label = "knight"
        max_this_class = 20
    elif(label == "uniquePawn"):
        label = "pawn"
        max_this_classS = 20
    elif(label == "uniquePlayer"):
        label = "player"
        max_this_class = 20
    elif(label == "uniqueRook"):
        label = "rook"
        max_this_class = 20
    
    files = files[:max_this_class] #limit amt of files

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
