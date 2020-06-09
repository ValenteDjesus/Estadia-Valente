#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:45:51 2020

Loading DataSet with vectors of image histograms and Initial Analysis of image histograms from MESSIDOR or APTOS

@author: generaviles
"""

import os
import easygui as gui
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

workingDir = gui.diropenbox(msg = 'Selecciona el directorio de trabajo.')
os.chdir(workingDir)

#datNorm = pd.read_csv('results/histNorm_MESSIDOR.csv')
datNorm = pd.read_csv('results/histTorax.csv')
#datPato = pd.read_csv('results/histPato_MESSIDOR.csv')
#datPato = pd.read_csv('results/histPato_APTOS.csv')


##############################################################################
#Generating an Average Representation of Normal and Pathological Histograms
##############################################################################

## Imágenes Normales
normMean = []
for i in range(7,255):
    normMean.append(round(np.mean(datNorm.iloc[:,i]), 0))

normMean = np.asarray(normMean)
normMean = np.pad(normMean, pad_width = (7,1))

np.savetxt('results/normMeanCOVID.csv', normMean, delimiter = ',')
#np.savetxt('results/normMeanMessidor.csv', normMean, delimiter = ',')

## Imágenes Patológicas
'''patoMean = []
for i in range(7,255):
    patoMean.append(round(np.mean(datPato.iloc[:,i]), 0))

patoMean = np.asarray(patoMean)
patoMean = np.pad(patoMean, pad_width = (7,1))

np.savetxt('results/patoMeanAPTOS.csv', patoMean, delimiter = ',')
#np.savetxt('results/patoMeanMessidor.csv', patoMean, delimiter = ',')
'''
## Visualizing both histograms

plt.subplot(1,1,1)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMean[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title('Histograma Promedio \n de Imágenes de Torax')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,1,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMean[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title('Histograma Promedio \n de Imágenes Patológicas')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''
plt.tight_layout()
plt.show()


######################################################################################################
#Generating a Representation with the Median of Normal and Pathological Histograms
######################################################################################################

## Imágenes Normales
normMedian = []
for i in range(7,255):
    normMedian.append(round(np.median(datNorm.iloc[:,i]), 0))

normMedian = np.asarray(normMedian)
normMedian = np.pad(normMedian, pad_width = (7,1))

np.savetxt('results/normMedianCOVID.csv', normMedian, delimiter = ',')
#np.savetxt('results/normMedianMessidor.csv', normMedian, delimiter = ',')

## Imágenes Patológicas
'''
patoMedian = []
for i in range(7,255):
    patoMedian.append(round(np.median(datPato.iloc[:,i]), 0))

patoMedian = np.asarray(patoMedian)
patoMedian = np.pad(patoMedian, pad_width = (7,1))

np.savetxt('results/patoMedianAPTOS.csv', patoMedian, delimiter = ',')
#np.savetxt('results/patoMedianMessidor.csv', patoMedian, delimiter = ',')
'''
## Visualizing both histograms

plt.subplot(1,1,1)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMedian[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title('Histograma Media \n de Imágenes de Torax')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,1,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMedian[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title('Histograma Media \n de Imágenes Patológicas')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''
plt.tight_layout()
plt.show()


######################################################################################################
# Keeping only values within 2 Standard Deviations
######################################################################################################

# Normal images
#mean = round(np.mean(datNorm.iloc[:,12]), 0)
#sd = round(np.std(datNorm.iloc[:,12]), 0)
#len([x for x in datNorm.iloc[:,15] if (x > mean - 1.5 * sd)])

newDatNorm = np.full((datNorm.shape[0],256),0)
#newDatNorm = pd.DataFrame(newDatNorm)

for i in range(0,len(datNorm.iloc[:,0])):
    for j in range(0,len(datNorm.iloc[0,:])):
        mean = round(np.mean(datNorm.iloc[:,j]), 0)
        sd = round(np.std(datNorm.iloc[:,j]), 0)
        if (datNorm.iloc[i,j] < (mean + 2 * sd) and datNorm.iloc[i,j] > (mean - 2 * sd)):
            newDatNorm[i,j] = datNorm.iloc[i,j]
        else:
            newDatNorm[i,j] = 0

newDatNorm = pd.DataFrame(newDatNorm)

#plt.plot(newDatNorm[:,150])

# Pathological Images
'''
newDatPato = np.full((datPato.shape[0],256),0)
#newDatNorm = pd.DataFrame(newDatNorm)

for i in range(0,len(datPato.iloc[:,0])):
    for j in range(0,len(datPato.iloc[0,:])):
        mean = round(np.mean(datPato.iloc[:,j]), 0)
        sd = round(np.std(datPato.iloc[:,j]), 0)        
        if (datPato.iloc[i,j] < (mean + 2 * sd) and datPato.iloc[i,j] > (mean - 2 * sd)):
            newDatPato[i,j] = datPato.iloc[i,j]
        else:
            newDatPato[i,j] = 0
newDatPato = pd.DataFrame(newDatPato)
'''
#plt.plot(datPato.iloc[:,150])
#plt.plot(newDatPato[:,150])

######################################################################################################
# Visualizing resulting histograms with data within 2 standard deviations
######################################################################################################

## Imágenes normales dentro de 2 desviaciones estándar
normMean2 = []
for i in range(7,255):
    normMean2.append(round(np.mean(newDatNorm.iloc[:,i]), 0))

normMean2 = np.asarray(normMean2)
normMean2 = np.pad(normMean2, pad_width = (7,1))

'''
## Imágenes Patológicas dentro de 2 desviaciones estándar
patoMean2 = []
for i in range(7,255):
    patoMean2.append(round(np.mean(newDatPato.iloc[:,i]), 0))

patoMean2 = np.asarray(patoMean2)
patoMean2 = np.pad(patoMean2, pad_width = (7,1))
'''

## Visualizing both histograms
plt.subplot(1,1,1)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMean2[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title(r'$\overline{X}$ Img de Torax, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,1,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMean2[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title(r'$\overline{X}$ Img Patológicas, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''
plt.tight_layout()
plt.show()

################################################################################

## Imágenes Normales
normMedian2 = []
for i in range(7,255):
    normMedian2.append(round(np.median(newDatNorm.iloc[:,i]), 0))

normMedian2 = np.asarray(normMedian2)
normMedian2 = np.pad(normMedian2, pad_width = (7,1))

## Imágenes Patológicas
'''
patoMedian2 = []
for i in range(7,255):
    patoMedian2.append(round(np.median(newDatPato.iloc[:,i]), 0))

patoMedian2 = np.asarray(patoMedian2)
patoMedian2 = np.pad(patoMedian2, pad_width = (7,1))
'''
## Visualizing both histograms

plt.subplot(1,1,1)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMedian2[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title(r'$mediana$ Img de Torax, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,1,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMedian2[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title(r'$mediana$ Img Patológicas, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''
plt.tight_layout()
plt.show()


#########################################################################################
## Plotting all two  histograms in one image
#########################################################################################

#MEAN

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2,2,1)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMean[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title(r'$\overline{X}$ Img de Torax')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,2,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMean[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title(r'$\overline{X}$ Img Patológicas')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''

plt.subplot(2,2,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMean2[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title(r'$\overline{X}$ Img de Torax, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,2,4)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMean2[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title(r'$\overline{X}$ Img Patológicas, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''
plt.tight_layout()
plt.savefig('results/meanCOVID.png')
plt.show()


#MEDIAN
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(2,2,1)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMedian[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title(r'$median$ Img de Torax')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,2,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMedian[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title(r'$median$ Img Patológicas')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''
plt.subplot(2,2,2)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),normMedian2[mini:maxi], color = (0.2, 0.4, 0.6, 0.7))
plt.title(r'$median$ Img de  Torax, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')

'''
plt.subplot(2,2,4)
mini = 0
maxi = 256
plt.bar(np.arange(mini,maxi, step = 1),patoMedian2[mini:maxi], color = (0.8, 0.2, 0.6, 0.6))
plt.title(r'$median$ Img Patológicas, valores dentro de 2$\sigma$')
plt.xlabel('Profundidad de Luminiscencia')
plt.ylabel('Conteo')
'''
plt.tight_layout()
plt.savefig('results/medianCOVID.png')
plt.show()
