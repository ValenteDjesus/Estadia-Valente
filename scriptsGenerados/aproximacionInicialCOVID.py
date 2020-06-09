#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:26:04 2020

@author: generaviles
"""

import os
#import matplotlib.pyplot as plt
#import functions as fn
import cv2 as cv
import numpy as np
#from keras import preprocessing as pp
import easygui as gui
import pandas as pd

'''
#################
#Reading into RGB
#################
#img = cv.imread('imgPat/20051020_44843_0100_PP.tif',-1)
img = cv.imread('imgPat/20051020_44901_0100_PP.tif',-1)
#img = cv.imread('imgPat/20051020_45004_0100_PP.tif',-1)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#Displaying image
plt.imshow(img)
plt.axis('off')
plt.show()

##################################
#Converting from RGB to Grayscale
##################################
imgGray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#Displaying image
plt.imshow(imgGray, cmap = 'gray')
plt.axis('off')
plt.show()

print('Min: %.3f, Max: %.3f' % (imgGray.min(), imgGray.max()))
##################################
#Image Histogram
##################################
mini = 7
imgHist = cv.calcHist([imgGray], [0], None, [256], [0,256])
plt.bar(np.arange(mini,256, step = 1),imgHist[mini:256,0])
plt.title('Histograma de Imagen')
plt.xlabel('Valores de Luminiscencia')
plt.ylabel('Conteo')
plt.show()
'''
#######################################################3
####    Iterating over files to generate datasets
#######################################################3

workingDir = gui.diropenbox(msg = 'Selecciona el directorio de trabajo.')
os.chdir(workingDir)

##Creatingf a gui for selecting files
inputdir = gui.diropenbox(msg = 'Selecciona el Folder Donde Están las Imágenes a Procesar')
#outdir = gui.diropenbox(msg = 'Selecciona el Folder a Donde Se Guardará la Base de Datos Resultante')
test_list = [ f for f in  os.listdir(inputdir)]

histsDataset = pd.DataFrame(index = range(0), columns = range(256))



for f in test_list:
    img = cv.imread(inputdir +'/'+ f,-1) #reading color image
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2RGB) #converting color to grayscale image
    print('Imagen %s: Valor Mínimo %.0f, Valor Max: %.0f' % (f,imgGray.min(), imgGray.max())) #Reporting status to terminal
    imgHist = cv.calcHist([imgGray], [0], None, [256], [0,256])
    imgHist = pd.DataFrame.from_records(imgHist)
    #imgEq = cv2.equalizeHist(img)
    print('Histograma ya salió para imagen %s y agregado a dataset.' % (f))
    histsDataset = histsDataset.append(imgHist.T) #Appending vector of values to dataset


#SAVING DataFrame to CSV
label = 'Torax'
print('Resultados exportados a csv')
histsDataset.to_csv (r'results/hist' + label + '.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path


'''
#Normalizing values
# PENDIENTE SACAR LOS VALORES BIEN Y GRAFICARLOS
imgPixels = imgGray.astype('float32')
imgPixels /= 255.0
print('Min: %.3f, Max: %.3f' % (imgPixels.min(), imgPixels.max()))
imgHistNorm = cv.calcHist([imgPixels], [0], None, [256], [0,1])
miniNorm = 7
pasos = 1/255
plt.bar(np.arange(0,1, step = 0.00392156862745098),imgHistNorm[0:256,0])
plt.title('Histograma de Imagen')
plt.xlabel('Valores de Luminiscencia Normalizados a Rango 0-1')
plt.ylabel('Conteo')
plt.show()
plt.bar(x, height, kwargs)
'''
