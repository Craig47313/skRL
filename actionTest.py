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

'''print("tile states: ")
print(te.tileStates)'''

possibleActions = te.getActions()
'''allActions = ""
for i in range(len(possibleActions)):
    allActions += (str(i) + " " + str(possibleActions[i]) + " | ")
print(allActions)'''

def clickAll():
    for state in range(1, 129):
        print(str(state))
        if(state < 65 and state >= 1):#moving
            x = ((state-1) % 8)
            y = 7-((state-1) // 8)
            #print(f"x, y: {x}, {y}")
            pyautogui.click(478+(x*73), 730 - (y*73))
            #pyautogui.mouseDown()
            #pyautogui.mouseUp()

        else:
            x = ((state-65) % 8)
            y = 7-((state-65) // 8)
            #print(f"x, y: {x}, {y}")
            pyautogui.click(478+(x*73), 730 - (y*73))
            pyautogui.mouseDown()
            pyautogui.mouseUp()

        time.sleep(0.1) 

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
        if(key.char == 'a'):
            print('clicking') 
            clickAll()
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