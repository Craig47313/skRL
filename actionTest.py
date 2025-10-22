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

te = actions.actions()
'''
te.getState()

print("tile states: ")
print(te.tileStates)

te.getActions()
'''


waiting = True
def click(key):
    global waiting
    try:
        '''if(key.char == 'c'):
            print('clicking') 
            waiting = False'''
        '''deadKing = 30 if(detectWin() or self.peiceAmts[2] == 0) else 0
        deadPlayer = -50 if(detectDeath() or self.peiceAmts[5]==0) else 0'''
        deadKing = 30 if(detectWin()) else 0
        deadPlayer = -50 if(detectDeath()) else 0
        done = (deadKing != 0 or deadPlayer != 0)
        win = deadKing != 0
        if(done): 
            restart(win)
    except:
        print("error")
def detectWin(self, threshold = 500):
    imgCrop = self.img[1600: 1800, 1000:1900]
    winScreenCrop = self.winScreen[1600: 1800, 1000:1900]
    if(np.mean((imgCrop.astype("float") - winScreenCrop.astype("float")) ** 2)) < threshold:#if mse b/t win img and img is the same then infers a win has happened
        return True
    else:
        return False   
def detectDeath(self, threshold = 500):
    if(np.mean((self.img.astype("float") - self.gameOverScreen.astype("float")) ** 2)) < threshold:#if mse b/t death img and img is the same then infers is a death
        return True
    else:
        self.alive = False
        return False 
def restart(didWin):
    if(didWin):#restart after a win
        print('clicking') 
        pyautogui.mouseDown()#quicken the pieces breaking
        pyautogui.mouseUp()
        time.sleep(2.0)

        pyautogui.click(600, 510)#click a modifier
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        time.sleep(2.0)

        pyautogui.mouseDown()#quicken the pieces forming
        pyautogui.mouseUp()
        time.sleep(2.0)

        pyautogui.keyDown('esc')#go into menu
        pyautogui.keyUp('esc')

        pyautogui.click(600, 720)#resign
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        pyautogui.mouseDown()
        pyautogui.mouseUp()

        pyautogui.mouseDown()#quicken resign screen
        pyautogui.mouseUp()

        pyautogui.click(600, 510)#click try again button
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        time.sleep(1.0)

        pyautogui.mouseDown()#quicken pieces reforming
        pyautogui.mouseUp()
        time.sleep(2.0)
    else:#restart after a loss
        pyautogui.click(600, 510)
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        time.sleep(1.0)
        pyautogui.mouseDown()
        pyautogui.mouseUp()
        time.sleep(1.0)
                
listener = keyboard.Listener(on_press=click)
listener.start()
while waiting:
    time.sleep(0.1)
#print((730-219)/7.0)