import os
import shutil
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

RAW_FOLDER = "dataraw"
DATASET_FOLDER = "data"

labels = ["board", "pawn", "player","rook", "knight", "bishop", "queen", "king", "na"]
for lbl in labels:
    os.makedirs(os.path.join(DATASET_FOLDER, lbl), exist_ok=True)

# Load images
images = [f for f in os.listdir(RAW_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
current = 0

root = tk.Tk()
root.title("Chess Labeling Tool")
numLabel = tk.Label(root, text= "lable loading")
numLabel.config(font=("ariel", 14))
numLabel.pack()

nameOfLastFile = "n/a"

img_label = tk.Label(root)
img_label.pack()

def show_image():
    global current, photo, nameOfLastFile
    if current < len(images):
        path = os.path.join(RAW_FOLDER, images[current])
        img = Image.open(path).resize((300, 300))  # resize for display
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo)
        numLabel.config(text=nameOfLastFile)
        nameOfLastFile = images[current]
    else:
        img_label.config(text="Fin")

def label_image(label):
    global current
    if current < len(images):
        src = os.path.join(RAW_FOLDER, images[current])
        dst = os.path.join(DATASET_FOLDER, label, images[current])
        shutil.move(src, dst)  # move file
        current += 1
        show_image()

# Buttons
for lbl in labels:
    tk.Button(root, text=lbl, command=lambda l=lbl: label_image(l)).pack(side=tk.LEFT)

show_image()
root.mainloop()
