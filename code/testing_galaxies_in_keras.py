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

#install packages
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0

#load data
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

r = '_r'
dotr = '.r'
#separate u and r band for SL
for fitsName in fitsNames_SL:
    if r in fitsName:
        
    else:
        del data_SL
r_SL = np.array(r_SL)

r_random = []
#separate u and r band for random galaxies
for fitsName in fitsNames_random :
    if dotr in fitsName:
        r_random.append(temp_random[fitsNames_random== fitsName])

r_random = np.array(r_random)

#Follow steps KERAS website-----------------------------------------------------
'''
model = EfficientNetB0(weights='imagenet')
#This model takes input images of shape (224, 224, 3), and the input data should range [0, 255].

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224

#example dataset: Stanford Dogs
batch_size = 64

dataset_name = "stanford_dogs"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train", "test"], with_info=True, as_supervised=True
)
NUM_CLASSES = ds_info.features["label"].num_classes
'''