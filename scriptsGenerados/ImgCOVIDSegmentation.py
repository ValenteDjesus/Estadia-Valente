# -*- coding: utf-8 -*-
"""

@author: Valente de Jesús López Reyes
Universidad Politécnica de Pachuca 
Ingeniería en Software
"""


import os
import easygui as gui
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sp
from sklearn import preprocessing
import normStandImg as ns
#%%

########################################################################
# Máscara sobre imagen en escala de grises
########################################################################

def maskGray(img,minValue,maxValue, backg=0):
    '''
    img: Image to create a mask from
    minValue: Minimmum value of the mask range
    maxValue: Maximmum value of the mask range
    '''
    #img = mean_gray_values
    
    minVal = np.min(img)
    maxVal = np.max(img)
    #
    X,Y = img.shape
    
    #mask = np.full((M,N),255, dtype = np.uint8)
    mask = np.zeros((X,Y), dtype=np.uint8)
    #mask = np.float32(mask)
    
    for i in range(X):
        for j in range(Y):
            if (img[i][j] < maxValue and img[i][j] > minValue):
                mask[i][j] = img[i][j]
            elif(backg == 1):
                mask[i][j]=255
            #Drawing  a line dividing the image
            #mask[i][int(Y/2)] = 255
    plt.imshow(mask,cmap='gray',vmax= np.max(img),vmin= np.min(img))
    plt.title('Mask obtained from GrayScale Img, range: ' + str(minValue)+', '+str(maxValue), fontsize = 'large', fontweight = 'bold')
    #plt.axis('off')
    plt.show()
    return mask

#Function to find peaks and apply smoothing to histograms of an image

def savitskyGolay(hist):
    peaks, _ = sp.find_peaks(hist,height=0)
    prominences = sp.peak_prominences(hist,peaks)[0]
    peaks, _ = sp.find_peaks(hist, prominence = np.median(prominences))
    #Finding peaks in image
    print('Number of peaks ', peaks.size)
    
    #obtaining Savitsky - golay window size with Nyquist - Shannon theorem
    #SC winSize = data points/lobes in data profile
    SGwindowSize = round(256/peaks.size)
    if(SGwindowSize%2 == 0):
        SGwindowSize+=1
    print('Window Size = ', SGwindowSize)
    
    #Applying Savitsky - golay filter
    #Using  the calculated SG Window size and a polynomial order of 1
    SGfilter = sp.savgol_filter(hist,SGwindowSize,polyorder = 1)
    return SGfilter
    
#%%
#Initializing working directories and the directory containing the images
workingDir = gui.diropenbox(msg = 'Selecciona el directorio de trabajo.')
os.chdir(workingDir)
inputdir = gui.diropenbox(msg = 'Selecciona el Folder Donde Están las Imágenes a Procesar')

#%%
#############################
#Reading images in grayscale
#############################
listImg = []
test_list = [ f for f in  os.listdir(inputdir)]
listHist = pd.DataFrame(index = range(0), columns = range(256))

print('Generating histograms...')
for f in range(0,len(test_list)):
    imgGray = cv.imread(inputdir +'/'+ test_list[f],0) #reading color image as grayscale
    #imgGray = cv.resize(imgGray, (1440,960))
    listImg.append(imgGray)
    imgHist = cv.calcHist([imgGray], [0], None, [256], [0,256])
    imgHist = pd.DataFrame.from_records(imgHist)
    listHist = listHist.append(imgHist.T)
    print("Img " ,f, "loaded and  histogram generated")

#Saving to CSV
label = 'Torax'
listHist.to_csv (r'results/hist' + label + '.csv', index = None, header=True)
print('Histograms saved to csv')

#%%
#Defining a sample image
working_img = listImg[2]
#Printing orginal image and its histogram for testing
plt.title("Original image")
plt.imshow(working_img, cmap = 'gray')
plt.show()
#%%
plt.title("Histogram from image")
plt.plot(listHist.iloc[2])
plt.show()


#%%
#########3############################################
#Normalizing images and obtaining histograms from them
######################################################

listNormHist = pd.DataFrame(index = range(0), columns = range(0,255))

for i in range(0,len(listImg)):
    imgNorm = ns.imgNorm(listImg[i])
    normHist,normValues = np.histogram(imgNorm.ravel(), bins=256, range=(np.min(imgNorm), np.max(imgNorm)))
    normHist = pd.DataFrame(normHist)
    listNormHist = listNormHist.append(normHist.T)
    print('Normalized histogram for img ',i,'generated')
    
normValues  = normValues[1:]
listNormHist.columns = normValues
#,_ = plt.hist(imgNorm.ravel(), bins=256, range=(np.min(imgNorm), np.max(imgNorm)), fc='k', ec='k')
plt.plot(listNormHist.iloc[4])
plt.show()
plt.imshow(imgNorm,cmap = 'gray')
plt.show()
#Saving to CSV
label = 'NormalizedTorax'
listNormHist.to_csv (r'results/hist' + label + '.csv', index = None, header=True)
print('Normalized histograms saved to csv')
#%%
##############################################
#Obtaining Savitzky Golay Filter for every Normalized histogram
############################################
SGhist = pd.DataFrame(index = range(0), columns = range(0,255))

for i in range(listNormHist.shape[0]):
    hist = listNormHist.iloc[i]
    smoothHist = savitskyGolay(hist)
    smoothHist = pd.DataFrame(smoothHist)
    SGhist = SGhist.append(smoothHist.T)
SGhist.columns = normValues
#Saving to CSV
label = 'SavitskyGolayTorax'
SGhist.to_csv (r'results/hist' + label + '.csv', index = None, header=True)
print('SG histograms saved to csv')
#%%
plt.title("Smoothed histogram from normalized image")
plt.plot(SGhist.iloc[8])
plt.show()
#Search for a delta with the difference of increment in the histograms
#%%
'''
#########3############################################
#Standardizing images and obtaining histograms from them
######################################################

listStandHist = pd.DataFrame(index = range(0), columns = range(0,255))


for i in range(0,len(listImg)):
    imgStand = ns.imgStand(listImg[i])
    standHist,standValues = np.histogram(imgStand.ravel(), bins=256, range=(np.min(imgStand), np.max(imgStand)))
    standHist = pd.DataFrame(standHist)
    listStandHist = listStandHist.append(standHist.T)
    print('Standardized histogram for img ',i,'generated')
listStandHist.columns = standValues
#,_ = plt.hist(imgNorm.ravel(), bins=256, range=(np.min(imgNorm), np.max(imgNorm)), fc='k', ec='k')
plt.plot(listStandHist.iloc[3])
plt.show()
plt.imshow(imgNorm,cmap = 'gray')
plt.show()
#Saving to CSV
label = 'StandardizedTorax'
SGhist.to_csv (r'results/hist' + label + '.csv', index = None, header=True)
print('standardized histograms saved to csv')
'''
    #%%
####Printing smoothed histograms
for i in range(SGhist.shape[0]):
    plt.title("Smoothed histogram from Normalized image "+str(i))
    plt.plot(SGhist.iloc[i])
    plt.show()
    
#%% 
#Defininfg a sample smoothed histogram
workImg = 52

data = np.array(SGhist.iloc[workImg])

#Obtaining local maxima 
maxIndex = np.array(sp.argrelextrema(data,np.greater)).T
maxVal = data[maxIndex] #obtaining luminiscence values from the  orginal histogram

plt.plot(maxIndex, data[maxIndex], "ob"); plt.plot(data); plt.legend(['maxima'])
plt.show()
print(maxIndex)

#Obtaining local minima UNUSED
minIndex = np.array(sp.argrelextrema(data,np.less)).T
plt.plot(minIndex, data[minIndex], "xk"); plt.plot(data); plt.legend(['minima'])
plt.show()

#Hacer un repositorio de google drive con iimagenes y mascaras.
#Actualizar github.
#Generar pseudocodigo de calculo de rangos y generar diagrama de flujo de todo el codigo

#%%

start = minIndex[0]


for i in range(maxIndex.size - 1): 
    if(maxVal[i+1] <=  maxVal[i] and maxIndex[i]>50):
        end = maxIndex[i]
        break

print(end)

#Obtener histogramas de todas las imagenes en excel
#Estandarizar imagenes primero
#obtener el intervalo deseado en la estandarizacion
#Aplicar el intervalo en la estandarizacion a los valores originales 
#Normalizar imagen media = 0, desviacion estandar = 0
#Oxigenacion vs volumen pulmnar
#Checar  region de luminiscncia al rededor de los pulmones
#%%
#Applying mask

plt.plot(listHist.iloc[workImg])
plt.show()
plt.imshow(listImg[workImg],cmap = 'gray')
plt.show()
maskimg = maskGray(listImg[workImg],start,end)
#cv.imwrite('C:/Users/USUARIO/Documents/Python projects/Working examples/mask.png', maskimg)
#%%
#Binarizing
binimg = cv.threshold(maskimg, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
plt.imshow(binimg,cmap = 'gray')
#%%
#Testing find contours from opencv
