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

model = EfficientNetB0(weights='imagenet')
#make include top false

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE =   224
NUM_CLASSES = 2 #strongly lensed vs not strongly lensed

#example dataset: Stanford Dogs, what do i choose now?
batch_size = 64

########################## galaxy data

# pick the header keys you want to dump to a table.
keys = ['NAXIS', 'FILTER'] #later this will be number of galaxies?
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
for i in range(len(fitsNames_SL)-1):
    data_SL[i,:,:] = data_SL[i,:,:]/np.max(data_SL[i,:,:])
    data_random[i,:,:] = data_random[i,:,:]/np.max(data_random[i,:,:])


################################################################################
#full dataset
data_full = np.concatenate((data_SL, data_random ))

def grayscale_to_rgb(images):
    # Stack grayscale images three times along the channel dimension to create RGB-like images
    rgb_images = np.stack([images, images, images], axis=-1)
    return rgb_images

#convert to rgb
data_full = grayscale_to_rgb(data_full)

#convert to tensor
tensor_data = tf.convert_to_tensor(data_full)

#create labels
labels_arr = np.concatenate((np.ones(60), np.zeros(60)))
labels_tensor = tf.convert_to_tensor(labels_arr, dtype=tf.int32)

#create dataset
dataset = tf.data.Dataset.from_tensor_slices(tensor_data)
labels = tf.data.Dataset.from_tensor_slices(labels_tensor)

#zip data and labels together
zipped_data = tf.data.Dataset.zip((dataset, labels))


#shuffle and batch the dataset
# Shuffle the dataset
shuffle_buffer_size = 120  # Set to the number of samples for full shuffling
dataset = zipped_data.shuffle(buffer_size=shuffle_buffer_size)

#print to check
#for data, labels in dataset:
    # print( "Labels:", labels.numpy())

# Calculate the size of the dataset
dataset_size = len(list(dataset))

# Calculate the size of the training set (e.g., 70% of the data)
train_size = int(0.7 * dataset_size)

size = (IMG_SIZE,IMG_SIZE)
# Apply the preprocess_image function to each grayscale image in the dataset using .map()
dataset = dataset.map(lambda image, label: (tf.image.resize(image, size), label))

# Split the dataset into training and test sets
ds_train = dataset.take(train_size)
ds_test = dataset.skip(train_size)

#continuing the steps from earlier
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))

#visualising the data
for i, (image, label) in enumerate(ds_train.take(9)):
    print(np.max(image))
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy())
    plt.title("{}".format(label.numpy()))
    plt.axis("off")

plt.show()

#trying to do ML #################################################################

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

# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

ds_train = ds_train.map(
    input_preprocess, num_parallel_calls=tf.data.AUTOTUNE
)
ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)

#this part doesn't work
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
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
'''
epochs = 40  # in paper 200, but keras can adjust it if necessary
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    #does not work:
    #plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    
plot_hist(hist)
'''