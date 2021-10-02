
import cv2
import numpy as np
import json
import os

import glob2 as glob

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Resize input images to 512X512 to fit SSD neural net
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


input_path = r'.\busesTrain'
os.chdir(input_path)
images=glob.glob("*.jpg")
BLACK = [0, 0, 0]

top = 0
bottom = int((3648-2736))
IMAGE_SIZE = 512

for imgStr in images:
    image = cv2.imread(imgStr)
    print(image.shape)
    dstImg = cv2.copyMakeBorder(image,top,bottom,0,0,cv2.BORDER_CONSTANT, value=BLACK)
    resized_image = cv2.resize(dstImg, (IMAGE_SIZE, IMAGE_SIZE))
    cv2.imwrite(imgStr[0:-4] + "_resized3.jpg" , resized_image)