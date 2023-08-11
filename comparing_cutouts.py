#import packages
from astropy import constants as const
from astropy import units as u
import csv
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
from astropy.io import fits
import glob
from astropy.io import fits as pyfits
from astropy.table import Table, Column

r = '_r'
dotr = '.r'

# pick the header keys you want to dump to a table.
keys = ['NAXIS', 'FILTER']
# pick the HDU you want to pull them from. It might be that your data are spectra, or FITS tables, or multi-extension "mosaics". 
hdu = 0

# get header keyword values
values_SL = []
fitsNames_SL = []
data_SL = []
for fitsName in glob.glob('./SL_cutouts/*.fits'):
    # opening the file is unnecessary. just pull the (right) header
    header = pyfits.getheader(fitsName, hdu)
    values_SL.append([header.get(key) for key in keys])
    
    #save data
    dat = pyfits.getdata(fitsName)
    data_SL.append(dat)
    
    #save filename
    fitsNames_SL.append(fitsName)

# get header keyword values
values_random = []
fitsNames_random = []
data_random = []
for fitsName in glob.glob('./andom_cutouts/*.fits'):
    # opening the file is unnecessary. just pull the (right) header
    header = pyfits.getheader(fitsName, hdu)
    values_random.append([header.get(key) for key in keys])
    
    #save data
    dat = pyfits.getdata(fitsName)
    data_random.append(dat)
    
    #save filename
    fitsNames_random.append(fitsName)

#convert to np.arrays
data_SL = np.array(data_SL)
data_random = np.array(data_random)

#set all negative pixels to zero
data_SL[data_SL<0] =0
data_random[data_random<0] =0
        
#normalize the data
data_SL = data_SL/np.max(data_SL)
data_random = data_random/np.max(data_random)        
        
temp_SL = list(data_SL)
temp_random = list(data_random)

#total brightness
bins = np.linspace(0, 13, 100)
plt.hist(np.sum(np.sum(data_SL, axis = 0), axis = 1), bins,alpha = 0.5, label = 'data_SL') 
plt.hist(np.sum(np.sum(data_random, axis = 0), axis = 1), bins, alpha = 0.5, label = 'data_random') 
plt.legend(loc='upper right')
plt.title('Total brightness')
plt.show()

u_SL = []
r_SL = []

#separate u and r band for SL
for fitsName in fitsNames_SL:
    if r in fitsName:
        r_SL.append(temp_SL[fitsNames_SL== fitsName])
    else:
        u_SL.append(temp_SL[fitsNames_SL== fitsName])

u_SL = np.array(u_SL)
r_SL = np.array(r_SL)

u_random = []
r_random = []

#separate u and r band for random galaxies
for fitsName in fitsNames_random :
    if dotr in fitsName:
        r_random.append(temp_random[fitsNames_random== fitsName])
    else:
        u_random.append(temp_random[fitsNames_random== fitsName])

u_random = np.array(u_random)
r_random = np.array(r_random)

brightness_u_SL = np.sum(np.sum(u_SL, axis = 0), axis = 1)
brightness_r_SL = np.sum(np.sum(r_SL, axis = 0), axis = 1)
brightness_u_random =np.sum(np.sum(u_random, axis = 0), axis = 1)
brightness_r_random =np.sum(np.sum(r_random, axis = 0), axis = 1)

#compare brightness
bins = np.linspace(0, 30, 100)
plt.hist(brightness_u_SL, bins, alpha = 0.5, label = 'u_SL')
plt.hist(brightness_u_random, bins, alpha = 0.5, label = 'u_random')
plt.legend(loc='upper right')
plt.title('u-band')
plt.show()

bins = np.linspace(0, 30, 100)
plt.hist(brightness_r_SL, bins, alpha = 0.5, label = 'r_SL')
plt.hist(brightness_r_random, bins, alpha = 0.5, label = 'r_random')
plt.legend(loc='upper right')
plt.title('r band')
plt.show()
