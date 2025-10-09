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
import actions
from pynput import keyboard
print('imports working')

#te = actions.actions()

#te.getState()

#print(te.tileStates)

#te.getActions()
waiting = True
def click(key):
    global waiting
    if(key.char == 'c'):
        print('clicking')                   
        '''for i in range(30):
            print("y", i*10)
            #pyautogui.moveTo(880, 3*i, duration=1)'''
        
        '''pyautogui.click(478, 730)#bottom left corner
        time.sleep(1)
        pyautogui.click(990, 219)#top right corner'''

        for x in range(0,8):
            for y in range(0,8):
                pyautogui.click(478+(x*73), 730 - (y*73))
                time.sleep(0.5)
        
        '''pyautogui.mouseDown()
        pyautogui.mouseUp()'''
        waiting = False
        
listener = keyboard.Listener(on_press=click)
listener.start()
while waiting:
    time.sleep(0.1)
print((730-219)/7.0)