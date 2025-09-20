import pyautogui
import os
from datetime import datetime
import time
import cv2
import numpy as np
import torch
from torch.nn import nn
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
    def getState(self, savePath):
        screenshot = pyautogui.screenshot()
        #img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        img = cv2.imread("screenshot.png")
        tiles = self.createTiles(img)

        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        playerState = -1
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
    def detectDeath(self):
        if(np.mean((self.img.astype("float") - self.gameOverScreen.astype("float")) ** 2)) < 500:#if mse b/t death img and that img is the same then infers is a death
            return True
        else:
            return False
    def getActions(self):
        unavialableTiles = [[False for _ in range(8)] for _ in range(8)] 
        for i in range(8): 
            for j in range(8):
                tileType = self.tileStates[i][j]  
                # self.classes = (0 "bishop", 1 "board", 2 "king", 3 "knight", 4 "pawn", 5 "player", 6 "queen", 7 "rook")

                if tileType == 1:  # board 
                    continue
                else:
                    unavialableTiles[i][j] = True  #can't move to occupied

                if tileType == 0:  # bishop
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        x, y = i + dx, j + dy
                        while 0 <= x < 8 and 0 <= y < 8:
                            unavialableTiles[x][y] = True
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
                                unavialableTiles[x][y] = True

                elif tileType == 3:  # knight
                    for dx, dy in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                                (1, -2), (1, 2), (2, -1), (2, 1)]:
                        x, y = i + dx, j + dy
                        if 0 <= x < 8 and 0 <= y < 8:
                            unavialableTiles[x][y] = True

                elif tileType == 4:  # pawn 
                    if i - 1 >= 0 and j + 1 < 8:  # up-left
                        unavialableTiles[i-1][j+1] = True
                    if i + 1 < 8 and j + 1 < 8:  # up-right
                        unavialableTiles[i+1][j+1] = True

                elif tileType == 7:  # rook
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        x, y = i + dx, j + dy
                        while 0 <= x < 8 and 0 <= y < 8:
                            unavialableTiles[x][y] = True
                            if self.tileStates[x][y] != 1:  # stop if blocked
                                break
                            x += dx
                            y += dy

                elif tileType == 6:  # queen 
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1),
                                (1, 0), (-1, 0), (0, 1), (0, -1)]:
                        x, y = i + dx, j + dy
                        while 0 <= x < 8 and 0 <= y < 8:
                            unavialableTiles[x][y] = True
                            if self.tileStates[x][y] != 1:  # stop if blocked
                                break
                            x += dx
                            y += dy
                #no logic needed for player


        



    


    