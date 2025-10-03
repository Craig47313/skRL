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

class actions():
    def __init__(self):
        self.enemyPos = []

        net = models.resnet18(weights=None)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 8)  
        net.load_state_dict(torch.load("resnet_chess.pth", map_location=torch.device("cpu")))
        net.eval()
        self.net = net
        self.gameOverScreen = cv2.imread("gameOverScreen.png")
        self.classes = ("bishop", "board", "king", "knight", "pawn", "player", "queen", "rook")
        self.spareAmmo = 6
        self.currentAmmo = 2 
        self.tileStates = [[1 for _ in range(8)] for _ in range(8)]
    def createTiles(self, img):
        imgCrop = img[440: 1616, 880:2056]
        channelsAmt = 3
        tileWidth = (imgCrop.shape[0]//8)
        tileHeight = (imgCrop.shape[1]//8)

        tiles = imgCrop.reshape(8, tileWidth, 8, tileHeight, channelsAmt)
        tiles = tiles.transpose(0, 2, 1, 3, 4)
        tiles = tiles.reshape(64, tileWidth, tileHeight, channelsAmt)
        #print('shape of tiles when creating')
        #print(tiles[0].shape)
        return tiles
    def getState(self):
        screenshot = pyautogui.screenshot()
        #img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        img = cv2.imread("screenshot.png")

        self.img = img

        if(img is None):
            return -1
        
        tiles = self.createTiles(img)

        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        playerState = None
        tileStates = []

        flatIdx = -1
        for i in range(8):#64 tiles
            row = []
            for j in range(8):
                flatIdx += 1
                img = Image.fromarray(tiles[flatIdx].astype(np.uint8)) 
                img_t = transform(img).unsqueeze(0)  # batch dimension

                with torch.no_grad():
                    outputs = self.net(img_t)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    predInd = predicted.item()
                    label = self.classes[predInd]
                    if(label == "player"):
                        playerState = (i, j)
                row.append(predInd)
            tileStates.append(row)

        self.lastTiles = self.tileStates
        self.tileStates  =  tileStates
        self.playerState = playerState
        return np.array(tileStates).flatten()
    def detectDeath(self, threshold = 500):
        if(np.mean((self.img.astype("float") - self.gameOverScreen.astype("float")) ** 2)) < threshold:#if mse b/t death img and that img is the same then infers is a death
            return True
        else:
            return False
    def getActions(self, shootingDegree = 20):

        if(not (self.detectDeath() or self.playerState == None)):
            avialableTiles = [[True for _ in range(8)] for _ in range(8)] 
            for i in range(8): 
                for j in range(8):
                    tileType = self.tileStates[i][j]  
                    # self.classes = (0 "bishop", 1 "board", 2 "king", 3 "knight", 4 "pawn", 5 "player", 6 "queen", 7 "rook")

                    if tileType == 1:  # board 
                        continue
                    else:
                        avialableTiles[i][j] = False  #can't move to occupied

                    if tileType == 0:  # bishop
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            x, y = i + dx, j + dy
                            while 0 <= x < 8 and 0 <= y < 8:
                                avialableTiles[x][y] = False
                                if self.tileStates[x][y] != 1:  # stop if blocked
                                    break
                                x += dx
                                y += dy

                    elif tileType == 2:  # king
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                x, y = i + dx, j + dy
                                if 0 <= x < 8 and 0 <= y < 8:
                                    avialableTiles[x][y] = False

                    elif tileType == 3:  # knight
                        for dx, dy in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                                    (1, -2), (1, 2), (2, -1), (2, 1)]:
                            x, y = i + dx, j + dy
                            if 0 <= x < 8 and 0 <= y < 8:
                                avialableTiles[x][y] = False

                    elif tileType == 4:  # pawn 
                        if i - 1 >= 0 and j + 1 < 8:  # up-left
                            avialableTiles[i-1][j+1] = False
                        if i + 1 < 8 and j + 1 < 8:  # up-right
                            avialableTiles[i+1][j+1] = False

                    elif tileType == 7:  # rook
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            x, y = i + dx, j + dy
                            while 0 <= x < 8 and 0 <= y < 8:
                                avialableTiles[x][y] = False
                                if self.tileStates[x][y] != 1:  # stop if blocked
                                    break
                                x += dx
                                y += dy

                    elif tileType == 6:  # queen 
                        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1),
                                    (1, 0), (-1, 0), (0, 1), (0, -1)]:
                            x, y = i + dx, j + dy
                            while 0 <= x < 8 and 0 <= y < 8:
                                avialableTiles[x][y] = False
                                if self.tileStates[x][y] != 1:  # stop if blocked
                                    break
                                x += dx
                                y += dy
                
                    #elif (tileType == 5):  #no logic needed for player yet
                        
            
            

            allMoves = [[False for _ in range(8)] for _ in range(8)] 

            shootingMoves = np.zeros(360//shootingDegree, dtype=bool)
            degrees = [( np.round(np.cos(np.deg2rad(i*shootingDegree)), 2), np.round(np.sin(np.deg2rad(i*shootingDegree)), 2) ) for i in range(len(shootingMoves))]

            '''for i in range(len(shootingMoves)):
                print("degree:", i*shootingDegree, "cos:", degrees[i][0], "sin:", degrees[i][1])'''
            #print(degrees)
            #basic king movement:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    print(f"new dx{dx} dy{dy}")
                    if dx == 0 and dy == 0:
                        continue
                    x, y = self.playerState[0] + dx, self.playerState[1] + dy
                    if 0 <= x < 8 and 0 <= y < 8:
                        # shape (n,2)
                        allMoves[x][y] = True

                        degreeMask = np.zeros(360//shootingDegree, dtype=bool)
                        for i in range(360 // shootingDegree):
                            vx, vy = degrees[i]
                            # Compare signs of dx, dy with vx, vy
                            if np.sign(vx) == np.sign(dx) and np.sign(vy) == np.sign(dy):
                                degreeMask[i] = True
                        #print(degreeMask)
                        
                        shootingMoves = np.bitwise_or(degreeMask, shootingMoves)
                        #print(degreeMask)
            print(shootingMoves)
            print("player x, y: ", self.playerState[0], self.playerState[1])
           
            possibleMoves = np.bitwise_and(np.array(allMoves), np.array(avialableTiles)).flatten()
            print(possibleMoves)
            return np.concatenate((possibleMoves, shootingMoves))
        else:
            return -1

            



        



    


    