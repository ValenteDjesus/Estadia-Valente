#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:11:24 2020

Estandarización y normalización para arreglo de histogramas de imágen.
Script desarrollado por Gener Avilés-R para proyecto en colaboración con Valente.
https://www.statisticshowto.com/normalized/

@author: generaviles
"""
import os
import easygui as gui
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import cv2 as cv

#img = cv.imread('nejmoa2001191_f5-PA.jpeg',0)
#img = cv.imread('ryct.2020200034.fig5-day0.jpeg',0)
#plt.imshow(img,cmap='gray')

#Normalization
#workingDir = gui.diropenbox('Selecciona el directorio de trabajo de partida para el proyecto')
#os.chdir(workingDir)

def imgNorm(img):
    m,n = img.shape
    maxVal = np.max(img)
    minVal = np.min(img)
    imgNew = np.zeros(shape=(m,n), dtype= np.float32)
    for i in range(m):
        for j in range(n):
            imgNew[i][j] = (img[i][j] - minVal) / (maxVal - minVal)
    return imgNew

#imgNorm = imgNorm(img)
#plt.hist(imgNorm.ravel(), bins=256, range=(np.min(imgNorm), np.max(imgNorm)), fc='k', ec='k')
#plt.show()

#Standardization
def imgStand(img):
    m,n = img.shape
    mean = np.mean(img)
    sd = np.std(img)
    imgNew = np.zeros(shape=(m,n), dtype= np.float32)
    for i in range(m):
        for j in range(n):
            imgNew[i][j] = (img[i][j] - mean) / sd
    return imgNew

#imgStand = imgStand(img)
#plt.hist(imgStand.ravel(), bins=256, range=(np.min(imgStand), np.max(imgStand)), fc='k', ec='k')
#plt.show()