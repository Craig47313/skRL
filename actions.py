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
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        tiles = self.createTiles(img)

        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        ])
        states = []
        playerState = -1
        for i in range(64):#64 tiles
            img = Image.fromarray(tiles[i].astype(np.uint8)) 
            img_t = transform(img).unsqueeze(0)  # batch dimension

            with torch.no_grad():
                outputs = self.net(img_t)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                predInd = predicted.item()
                label = self.classes[predInd]
                if(label == "player"):
                    playerState = i
            states.append(predInd)
        self.lastImg = img
        self.states =  states
        self.playerState = playerState
    def detectDeath(self):
        if(np.mean((self.img.astype("float") - self.gameOverScreen.astype("float")) ** 2)) < 500:
            return True
        else:
            return False


    


    