import cv2
import numpy as np

winScreen = cv2.imread("winScreen.png")
lossScreen = cv2.imread("gameOverScreen.png")
img = cv2.imread("screenshot.png")
imgCrop = img[1600: 1800, 1000:1900]
winScreenCrop1 = winScreen[1600: 1800, 1000:1900]
winScreenCrop2 = winScreen[1600: 1800, 1000:1900]
lossCrop = lossScreen[1600: 1800, 1000:1900]
print(np.mean((img.astype("float") - lossScreen.astype("float")) ** 2))