import pyautogui
import cv2
import numpy as np
from pynput import keyboard
import os
import random 
import matplotlib.pyplot as plt

display = True #flag for wether to display the img or not

screenshot_done = False
show_window = False
save_img = False
runProgram = True
displayType = 'fullImg'

def on_press_display(key):
    global screenshot_done, show_window, save_img, runProgram
    try:
        if key.char == 's' and not screenshot_done:
            screenshot_done = True
            show_window = True
        elif key.char == 'r' and not screenshot_done:
            screenshot_done = True
            save_img = True
            show_window = False
        elif key.char == 'q':
            # stop the program by setting flag
            show_window = False
            runProgram = False
        elif key.char == 'e' and screenshot_done:
            show_window = False
            save_img = True
    except AttributeError:
        pass  # ignore special keys

def on_press_quicksave(key):
    global screenshot_done, show_window, save_img
    try:
        if key.char == 's' and not screenshot_done:
            screenshot_done = True
            save_img = True
            show_window = False
    except AttributeError:
        pass

def createTiles(img):
    imgCrop = img[440: 1616, 880:2056]
    channelsAmt = 3
    tileWidth = (imgCrop.shape[0]//8)
    tileHeight = (imgCrop.shape[1]//8)

    tiles = imgCrop.reshape(8, tileWidth, 8, tileHeight, channelsAmt)
    tiles = tiles.transpose(0, 2, 1, 3, 4)
    tiles = tiles.reshape(64, tileWidth, tileHeight, channelsAmt)
    print('shape of tiles when creating')
    print(tiles[0].shape)
    return tiles

def save_tiles(tiles, save_dir, screenshotId):
    for i in range(64):
        tile = tiles[i]
        fname = os.path.join(save_dir, f"tile_{i}_{screenshotId}.png")
        cv2.imwrite(fname, tile)
    print('shape of tiles when saving')
    print(tiles[0].shape)
    print("Screenshot id: " + screenshotId)
def show_tiles(tiles):
    # tiles.shape = (64, 147, 147, 3)
    rows, cols = 8, 8
    fig, axes = plt.subplots(rows, cols, figsize=(4, 4))

    for i, ax in enumerate(axes.flat):
        ax.imshow(tiles[i].astype(np.uint8))
        ax.axis("off")
    #plt.tight_layout()
    plt.show()

if(display):
    listener = keyboard.Listener(on_press=on_press_display)
else:
    listener = keyboard.Listener(on_press=on_press_quicksave)
listener.start()

print("Waiting for 's' key to capture screenshot...")

while runProgram:
    if screenshot_done:
        screenshot = pyautogui.screenshot()
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        tiles = createTiles(img)
        if(show_window):
            if(displayType == 'tile'):
                show_tiles(tiles)
            else:
                cv2.imshow('screenshot', img)
            while show_window:
                cv2.waitKey(100)
            plt.close('all')
            print("Window closed!")
        
        if(save_img):
            save_tiles(tiles, 'dataraw',str(random.random()))
        
        screenshot_done = False
        show_window = False
        save_img = False
