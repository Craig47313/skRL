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

'''te = actions.actions()

te.getState()

print("tile states: ")
print(te.tileStates)

te.getActions()
'''


waiting = True
def click(key):
    global waiting
    try:
        if(key.char == 'c'):
            print('clicking') 
            '''
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
            time.sleep(2.0)'''
            time.sleep(3.0)
            pyautogui.click(600, 510)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(1.0)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(1.0)
            '''shootingDegree = 15

            imgX = 4
            imgY = 0'''
            #for i in range(0, 360, shootingDegree):
                #pyautogui.click(478+(imgX*73), 730 - (imgY*73))
                #print(i)
                #pyautogui.click(478+(imgX*73) + 100*np.cos(np.deg2rad(i)), 730-(imgY*73) + 100*np.sin(np.deg2rad(i)))

                #pyautogui.mouseDown()
                #pyautogui.mouseUp()              
            #for i in range(30):
                #print("y", i*10)
                #pyautogui.moveTo(880, 3*i, duration=1)
            
            #pyautogui.click(478, 730)#bottom left corner
            #time.sleep(1)
            #pyautogui.click(990, 219)#top right corner

            #for x in range(0,8):
                #for y in range(0,8):
                    #pyautogui.click(478+(x*73), 730 - (y*73))
                    #time.sleep(0.5)
            
            #pyautogui.mouseDown()
            #pyautogui.mouseUp()
            waiting = False
    except:
        print("error")
        
listener = keyboard.Listener(on_press=click)
listener.start()
while waiting:
    time.sleep(0.1)
#print((730-219)/7.0)