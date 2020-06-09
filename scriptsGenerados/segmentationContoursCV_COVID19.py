# -*- coding: utf-8 -*-
"""
Created on Wed May 13 20:34:55 2020

@author: USUARIO
"""

import cv2
import numpy as np
import easygui as gui
import matplotlib.pyplot as plt

# Load image, grayscale, Otsu's threshold 
#inputdir = gui.diropenbox(msg = 'Selecciona el Folder Donde Están las Imágenes a Procesar')
image = cv2.imread('C:/Users/USUARIO/Documents/Python projects/Working examples/6.png',0)
plt.imshow(image, cmap = 'gray')
plt.title('Original image')
plt.show()

copyimg = image.copy()
#Reducing image scale
if(copyimg.shape[0]>2000 or copyimg.shape[1]>2000):
    copyimg = cv2.resize(copyimg, (0,0), fx=0.1, fy=0.1) 
else:
    copyimg = cv2.resize(copyimg, (0,0), fx=0.3, fy=0.3)
    
#Blurring image to  prevent finding too many contours
copyimg = cv2.blur(copyimg, (7,7))

original = copyimg.copy()
binimg = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

f, ax = plt.subplots(2,1)
ax[0].imshow(original, cmap='gray')
ax[0].set_title("Blurred and binarized imgs")
ax[1].imshow(binimg, cmap='gray')
plt.show()

#%%
# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
ROI_list = []

cnts = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS   )
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(copyimg, (x, y), (x + w, y + h), (255,255,255), 2)
    ROI = original[y:y+h, x:x+w]
    #Saves the ROIs
    #cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    ROI_number += 1
    ROI_list.append(ROI)
    
plt.imshow(copyimg,cmap='gray')
plt.show()

gray_list = []
for c in ROI_list:
    plt.imshow(c, cmap='gray')
    plt.show()
    
#plt.imshow(image)
#Checar transformaciones morfologicas: erosion y dilatacion 
#Armar un iterador que erosione, calcule  contornos
#Obtener la ariacion en porcentaje de ambos  pulmones y respecto a los demas ROI
