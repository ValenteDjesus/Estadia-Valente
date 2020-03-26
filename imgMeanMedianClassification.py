#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:01:20 2020
@author: telematica
"""

import os
import easygui as gui
workingDir = gui.diropenbox(msg = 'Selecciona el directorio de trabajo donde esté la base de datos.')
os.chdir(workingDir)
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################################################
#
# Leyendo imágenes, transformándola a escala de grises y midiendo distancia de Wasserstine
# al histograma de imagen por mediana de imágenes NORMALES
#
############################################################################################

inputdir = gui.diropenbox(msg = 'Selecciona el Folder Donde Están las Imágenes a Procesar')
#outdir = gui.diropenbox(msg = 'Selecciona el Folder a Donde Se Guardará la Base de Datos Resultante')
#test_list = [ f for f in  os.listdir(inputdir)]

############################################################################################
#
# Obteniendo la mediana de los histogramas de imágenes NORMALES
#
############################################################################################
#inputdirMod = gui.diropenbox(msg = 'Selecciona el Folder Donde Están Imágenes a Procesar para Obtener HistImg Modelo')
histsDataset = pd.DataFrame(index = range(0), columns = range(256))
listImg = []
test_list = [ f for f in  os.listdir(inputdir)]

imgGray_all = cv.imread(inputdir +'/'+ test_list[0] ,0)
imgGray_all = cv.resize(imgGray_all, (1440,960))#, interpolation = cv.INTER_AREA))
listImg.append(imgGray_all)
print('Generando Histogramas de Imagen de Subconjunto a Comparar')
for f in range(1,len(test_list)):
    imgGray = cv.imread(inputdir +'/'+ test_list[f],0) #reading color image as grayscale
    imgGray = cv.resize(imgGray, (1440,960))
    listImg.append(imgGray)
    imgGray_all = np.dstack((imgGray_all,imgGray))
    print("Img " ,f, "processed")

label = gui.choicebox(msg = 'Selecciona la etiqueta adecuada para las imágenes procesadas.',title = 'Etiqueta para Imágenes', choices = ['Normales','Patológicas'])

median_gray_values = np.median(imgGray_all,axis=2)
mean_gray_values = np.mean(imgGray_all,axis=2)


#%%
## Visualizing both representative images

plt.imshow(median_gray_values, cmap = 'gray')

#%%
plt.imshow(mean_gray_values , cmap = 'gray')

#%%
#Trying to put both images int oone plot
f,ax= plt.subplots(2,1)
ax[0].set_title('Representative imgs from mean and median imgs')
ax[0].imshow(mean_gray_values, cmap = 'gray')
ax[1].imshow(median_gray_values, cmap = 'gray')

#plt.axis('off')
plt.show()
#%%
#Generating slices for interest points in mean representative Image
#sliceLeft = mean_gray_values[400:550 , 450:600]
#Leftmean = sliceLeft.mean()
#plt.imshow(sliceLeft, cmap = 'gray')
#print(Leftmean)

#%%
#Creating directories for images and classifying
os.chdir(workingDir + "/results")
if(not 'Leftabnormal' in os.listdir()):
    os.mkdir('Leftabnormal')
if(not 'Rightabnormal' in os.listdir()):
    os.mkdir('Rightabnormal')
#Categorizing images
for im,n in zip(listImg,range(len(listImg))):
    res = classifyImg(im, test_list[n])
    print("img ", test_list[n], "classified: ",res)

#%%
##########################################################################
    #Clasiffy left and right images
##########################################################################
def classifyImg(img, name):
    os.chdir(workingDir)
    X,Y = img.shape
    aux = maskGray(img)
    maxSliceLeft = aux[350:550 , :int(Y/2)].max()
    maxSliceRight = aux[350:550 , int(Y/2):].max()
    if(maxSliceLeft<=255 and maxSliceLeft>=90 and maxSliceLeft > maxSliceRight):
        #os.chdir(workingDir+ "/results/Leftabnormal")
        #cv.imwrite(str(name[:-4]) + "Left.tif", img)
        return("Left")
    else:
        #os.chdir(workingDir+ "/results/Rightabnormal")
        #cv.imwrite(str(name[:-4]) + "Right.tif", img)
        return("Right")
    
#%%

########################################################################
# Máscara sobre imagen en escala de grises
########################################################################
def maskGray(img):
    #img = mean_gray_values
    minValue = 90
    maxValue = 255
    
    minVal = np.min(img)
    maxVal = np.max(img)
    #
    X,Y = img.shape
    
    #mask = np.full((M,N),255, dtype = np.uint8)
    mask = np.zeros((X,Y), dtype=np.uint8)
    mask = np.float32(mask)
    
    for i in range(X):
        for j in range(Y):
            if (img[i][j] < maxValue and img[i][j] > minValue):
                mask[i][j] = img[i][j]
            #else:
                #mask[i][j]=255
            #mask[i][int(Y/2)] = 255
    plt.imshow(mask,cmap='gray',vmax= np.max(img),vmin= np.min(img))
    plt.title('Mask obtained from GrayScale Img', fontsize = 'large', fontweight = 'bold')
    plt.axis('off')
    plt.show()
    return mask