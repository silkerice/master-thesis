
#creating a dataset suitable for ranking from my simulations
#we assume each 'query' is correct


#import packages
from astropy import constants as const
from astropy import units as u
import csv
from matplotlib import pyplot as plt
import numpy as np
import os
from astropy.io import fits
import glob
from astropy.io import fits as pyfits
from astropy.table import Table, Column
#tensor packages
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from scipy.ndimage import zoom
import random

#import previous work
import ML_sim_trial as ML

filename_SL='./SL_sim_gal2/*.fits'
filename_random='./random_sim_gal/*.fits'
sample_size = 20

#create dataset
hdu = 0
keys = ['NAXIS', 'COMMENT']

def create_ranking_data(filename_SL, filename_random, sample_size, hdu = 0, keys = ['NAXIS', 'COMMENT'], n = 800):
    #read data
    data_SL, fitsNames_SL = ML.get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys
    data_random, fitsNames_random = ML.get_data_from_file(filename_random, hdu=hdu, keys=keys)
        
    data_random_new = []
    for i in range(len(data_random)):
        arr = np.array(data_random[i])
        if np.shape(arr) == (480,640,4):
            data_random_new.append(arr)
    
    #put together
    data = np.concatenate((np.array(data_SL)[:,:,:,0], np.array(data_random_new )[:,:,:,0])) 
    fitsNames = fitsNames_SL + fitsNames_random
    
    #sample
    #create array
    ranking_data = []
    
    for i in range(0,int(n/sample_size)): 
        #sample the names and generate the corresponding labels
        sampled_names= random.sample(fitsNames, sample_size)
        sampled_labels = ML.create_labels(sample_size, sampled_names)
        for j in range(0,sample_size):
            tupple = [(sampled_names[j],i, int(sampled_labels[j]))]
            ranking_data.append(tupple)

        
    return ranking_data, fitsNames
    
#now we have a list with  galaxy filename, query ID, label
data, fitsNames = create_ranking_data(filename_SL, filename_random, sample_size)
    








