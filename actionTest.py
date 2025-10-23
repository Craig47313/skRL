import pyautogui
import os
from datetime import datetime
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from actions import actions
from pynput import keyboard
print('imports working')
global te
te = actions(True)
te.getState()

print("tile states: ")
print(te.tileStates)

possibleActions = te.getActions()
allActions = ""
for i in range(len(possibleActions)):
    allActions += (i + " " + possibleActions[i] + " | ")
print(allActions)


global waiting
waiting = True

def click(key):
    global te
    global state
    global waiting
    try:
        if(key.char == 'c'):
            print('clicking') 
            te.act(state)
            waiting = False          
    except:
        pass

global state
state = int(input("enter state to test: "))
listener = keyboard.Listener(on_press=click)
listener.start()
while waiting:
    time.sleep(0.1)
#print((730-219)/7.0)