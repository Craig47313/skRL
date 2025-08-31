import cv2
import matplotlib.pyplot as plt
import numpy as np
print(cv2.__version__)
img = cv2.imread("screenshot.png")
imgCrop = img[440: 1616, 880:2056]
#plt.imshow(img)


channelsAmt = 3
tileWidth = (imgCrop.shape[0]//8)
tileHeight = (imgCrop.shape[1]//8)

tiles = imgCrop.reshape(8, (imgCrop.shape[0]//8), 8, (imgCrop.shape[1]//8), channelsAmt)
tiles = tiles.transpose(0, 2, 1, 3, 4)
tiles = tiles.reshape(64, (imgCrop.shape[0]//8), (imgCrop.shape[1]//8), channelsAmt)


'''
# tiles.shape = (64, 147, 147, 3)
rows, cols = 8, 8
fig, axes = plt.subplots(rows, cols, figsize=(4, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(tiles[i].astype(np.uint8))
    ax.axis("off")
#plt.tight_layout()
plt.show()
'''

print(tiles[0].shape)
#plt.imshow(tiles[0])
#plt.show()


#print(imgCrop.shape)