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

def make_data_positive(data):
    #data can be list or array
    #convert to np.arrays
    data= np.array(data)
    for i in range(len(data[:,0,0])):
        minimum= np.min(data[i,:,:])
        eps = 0.0001
        data[i,:,:]=data[i,:,:]+abs(minimum)+eps
        print(data[i,:,:])
    
    #set all negative pixels to zero
    #data[data<0] =0
    return(data)
            
def normalize_data(data):
    #data can be list or array
    #convert to np.arrays
    #data = np.array(data)
    #normalize the data
    for i in range(len(data[:,0,0] )-1):
        data[i,:,:] = data[i,:,:]/np.max(data[i,:,:])
    return(data)

def get_data_from_file(filename, hdu = 0, keys = ['NAXIS', 'FILTER']):
    #filename is a string, stronglens is 1 or 0
    # get header keyword values
    values = []
    fitsNames = []
    data = []

    for fitsName in glob.glob(filename):
        # opening the file is unnecessary. just pull the (right) header
        header = pyfits.getheader(fitsName, hdu)
        values.append([header.get(key) for key in keys])
        
        #save data
        dat = pyfits.getdata(fitsName)
        data.append(dat)
        
        #save filename
        fitsNames.append(fitsName)

    return(data, fitsNames)#could also return values of headers

def grayscale_to_rgb(images):
    # Stack grayscale images three times along the channel dimension to create RGB-like images
    rgb_images = np.stack([images, images, images], axis=-1)
    return rgb_images

def create_labels(n, fitsNames):
    #create labels for training data
    labels = np.zeros(n)
    
    #check if stronglens
    i = 0
    for fitsName in fitsNames:
        if 'SL' in fitsName:
            labels[i] = 1
            i = i+1
    return(labels)

def input_preprocess(image, label, NUM_CLASSES = 2):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label
    

def prepare_data(filename_SL, filename_random, IMG_SIZE, hdu = 0, keys = ['NAXIS', 'FILTER']):
    #filename is a string, stronglens is 1 or 0
    data_SL, fitsNames_SL = get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys
    data_random, fitsNames_random = get_data_from_file(filename_random, hdu=hdu, keys=keys)
    #put together
    data = np.concatenate((np.array(data_SL), np.array(data_random ))) 
    fitsNames = fitsNames_SL + fitsNames_random

    #normalize and make positive
    data = normalize_data(make_data_positive(data))

    #convert to rgb
    data = grayscale_to_rgb(data)
    
    #convert to tensor
    tensor_data = tf.convert_to_tensor(data)
    
    #create labels
    n = len(fitsNames)
    labels = create_labels(n, fitsNames)
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
    
    #create dataset
    dataset = tf.data.Dataset.from_tensor_slices(tensor_data)
    labels = tf.data.Dataset.from_tensor_slices(labels_tensor)
    
    #zip data and labels together
    zipped_data = tf.data.Dataset.zip((dataset, labels))
    
    # Shuffle the dataset
    shuffle_buffer_size = n  # Set to the number of samples for full shuffling
    dataset = zipped_data.shuffle(buffer_size=shuffle_buffer_size)
    
    # Calculate the size of the training set (e.g., 80% of the data)
    train_size = int(0.7 * n)
    
    # Apply the preprocess_image function to each grayscale image in the dataset using .map()
    size = (IMG_SIZE,IMG_SIZE)
    dataset = dataset.map(lambda image, label: (tf.image.resize(image, size), label))

    # Split the dataset into training and test sets
    ds_train = dataset.take(train_size)
    ds_test = dataset.skip(train_size)

    #continuing the steps from earlier
    ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
    ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
       
    return(ds_train, ds_test)

def visual_test(filename_SL, filename_random, IMG_SIZE):
    ds_train, ds_test = prepare_data(filename_SL, filename_random, IMG_SIZE)
    #visualising the data
    for i, (image, label) in enumerate(ds_train.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy())
        plt.title("{}".format(label.numpy()))
        plt.axis("off")

    plt.show()

#we use simple starting parameters
def perform_ML(filename_SL, filename_random, IMG_SIZE, hdu=0, keys = ['NAXIS', 'FILTER'], epochs = 40, batch_size = 64, NUM_CLASSES=2):
    #data augmentation
    img_augmentation = Sequential(
        [
            layers.RandomRotation(factor=0.15),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomFlip(),
            layers.RandomContrast(factor=0.1),
        ],
        name="img_augmentation",
    )
    
    #get training data set from prepare_data()
    ds_train, ds_test = prepare_data(filename_SL, filename_random, IMG_SIZE)
    
    ds_train = ds_train.map(
        input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(input_preprocess)
    ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=False)#
    
    cardinality = tf.data.experimental.cardinality(ds_test).numpy()
    
    if cardinality> 0:
        print("Data is non-empty")
    else:
        print("Data is empty")

    #assuming no tpu connection
    strategy = tf.distribute.MirroredStrategy()
    
    #training the model
    with strategy.scope():
        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = img_augmentation(inputs)
        outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    model.summary()

    epochs = 50  # in paper 200, but keras can adjust it if necessary
    hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)

    def plot_hist(hist):
        plt.plot(hist.history["accuracy"])
        #does not work:
        plt.plot(hist.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()
        
    plot_hist(hist)
    

#tests
filename_SL='./SL_cutouts/*.fits'
filename_random='./andom_cutouts/*.fits'
IMG_SIZE = 224
visual_test=visual_test(filename_SL, filename_random, IMG_SIZE)

#perform_ML(filename_SL, filename_random, IMG_SIZE)

    
    
    
    
    