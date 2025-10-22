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
        self.winScreen = cv2.imread("winScreen.png")
        self.classes = ("bishop", "board", "king", "knight", "pawn", "player", "queen", "rook")
        self.spareAmmo = 6
        self.currentAmmo = 3
        self.maxAmmo = 3
        self.maxAmmoSpare = 6
        self.tileStates = None
        self.peiceAmts = None
        self.alive = True
        self.actionSize = 1 + 64 + 64 #reload, move, shoot
        self.stateSize = 1 + 1 + 64 #num bullets in shotgun and num remaining + state of all the tiles
        self.state = None
        self.lastState = None
        self.getState() 

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
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        #img = cv2.imread("screenshot.png")

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
        amts = [0]*8
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
                    #print(predInd)
                    label = self.classes[predInd]
                    amts[predInd]+=1
                    if(label == "player"):
                        print("player at ", i, j)
                        playerState = (i, j)
                row.append(predInd)
            tileStates.append(row)
        self.lastPieceAmts = self.peiceAmts
        self.peiceAmts = amts
        self.lastTiles = self.tileStates
        self.tileStates  =  tileStates
        self.playerState = playerState
        self.lastState = self.state
        self.state = np.concatenate((self.currentAmmo, self.spareAmmo, np.array(tileStates).flatten()))

        return self.state
    def getReward(self):
        #remember ("bishop", "board", "king", "knight", "pawn", "player", "queen", "rook")
        dBishops = self.lastPieceAmts[0] - self.peiceAmts[0]
        deadKing = 30 if(self.detectWin() or self.peiceAmts[2] == 0) else 0
        deadPlayer = -50 if(self.detectDeath() or self.peiceAmts[5]==0) else 0
        dKnights = self.lastPieceAmts[3] - self.peiceAmts[3]
        dPawns = self.lastPieceAmts[4] - self.peiceAmts[4]
        dQueens = self.lastPieceAmts[6] - self.peiceAmts[6]
        dRooks = self.lastPieceAmts[7] - self.peiceAmts[7]

        done = (deadKing != 0 or deadPlayer != 0)
        win = deadKing != 0

        return (dPawns + dBishops*3 + dKnights*3 + dRooks*5 + dQueens*9 + deadKing + deadPlayer - 3), done, win
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
    def act(self, state):
        if(not self.Actions[state]):
            return -1
        if(state == 0):
            pyautogui.press('space')
            if(self.currentAmmo < self.maxAmmo and self.spareAmmo > 0):
                self.currentAmmo+=1
                self.spareAmmo-=1
            elif(self.currentAmmo == self.maxAmmo and self.spareAmmo < self.maxAmmoSpare):
                self.spareAmmo+=1
        elif(state < 65 and state > 1):#moving
            x = state // 8
            y = state % 8
            pyautogui.click(478+(x*73), 730 - (y*73))
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            if(self.currentAmmo < self.maxAmmo and self.spareAmmo > 0):
                self.currentAmmo+=1
                self.spareAmmo-=1
            elif(self.currentAmmo == self.maxAmmo and self.spareAmmo < self.maxAmmoSpare):
                self.spareAmmo+=1
            time.sleep(0.5)
        else:
            x = (state-65) // 8
            y = (state-65) % 8
            pyautogui.click(478+(x*73), 730 - (y*73))
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            time.sleep(0.5) 
        return 0
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
            

            
    def getActions(self):
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

            shootingMoves = [[True for _ in range(8)] for _ in range(8)] 

            #basic king movement:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    #print(f"new dx{dx} dy{dy}")
                    if dx == 0 and dy == 0:
                        continue
                    x, y = self.playerState[0] + dx, self.playerState[1] + dy
                    if 0 <= x < 8 and 0 <= y < 8:
                        # shape (n,2)
                        allMoves[x][y] = True
            for x in range(8):
                for y in range(8):
                    if(allMoves[x][y]):#cant shoot if you can move to a tile
                        shootingMoves[x][y] = False
                    if(self.tileStates[x][y] != 1 and self.tileStates[x][y] != 2):#override if there is a piece on that tile
                        shootingMoves[x][y] = True


            #print("shooting moves: ")
            #print(shootingMoves)
            #print("player x, y: ", self.playerState[0], self.playerState[1])
           
            possibleMoves = np.bitwise_and(np.array(allMoves), np.array(avialableTiles)).flatten()
            #print("possible moves")
            #print(possibleMoves)

            canReload = self.spareAmmo > 0 and self.currentAmmo < self.maxAmmo
            reload = np.zeros(1, dtype=bool)
            if(canReload):
                reload = np.ones(1, dtype=bool)
            self.Actions = np.concatenate([reload, possibleMoves, np.array(shootingMoves).flatten()])
            return self.Actions#first state is reload (spacebar)
        else:
            return -1
    def step(self, action):
        self.act(action)
        nextState = self.getState()
        reward, done, win = self.getReward()
        if(done):
            self.restart(win)
        return reward, done, nextState
        




            



        



    


    