

import time

# Record the start time
start_time = time.time()

#import packages
from astropy import constants as const
from astropy import units as u
import csv
import cv2
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

from skimage.transform import resize

from sklearn.metrics import roc_curve, auc

# Function to apply histogram equalization
def equalize_histogram(image):
    image_np = image.numpy()  # Convert from tensor to numpy
    # Ensure image is in the range [0, 255] and type uint8
    image_np = (image_np * 255).astype(np.uint8)
    # Convert to grayscale if the image is not already in grayscale
    if len(image_np.shape) == 3 and image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    equalized_image = cv2.equalizeHist(image_np)
    return equalized_image

def make_data_positive(data):
    #data can be list or array
    #convert to np.arrays
    data= np.array(data)
    for i in range(len(data[:,0,0])):
        minimum= np.min(data[i,:,:])
        eps = 0.0001
        data[i,:,:]=data[i,:,:]+abs(minimum)+eps
    
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
    size = (IMG_SIZE,IMG_SIZE)
    
    #filename is a string, stronglens is 1 or 0
    data_SL, fitsNames_SL = get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys
    data_random, fitsNames_random = get_data_from_file(filename_random, hdu=hdu, keys=keys)
    
    #put together
    data = np.concatenate((np.array(data_SL), np.array(data_random ))) 
    fitsNames = fitsNames_SL + fitsNames_random
    print('concetanate successful')
    #normalize and make positive
    data = normalize_data(make_data_positive(data))

    #convert to rgb
    data = grayscale_to_rgb(data)
    
    #convert to tensor
    tensor_data = tf.convert_to_tensor(data)
    
    #create labels
    n = len(fitsNames)
    nsl = len(fitsNames_SL)
    print(n-nsl)
    print(nsl)
    #labels = create_labels(n, fitsNames)
    labels = np.concatenate((np.ones(nsl), np.zeros(n-nsl)))
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
    
    # Visualizing the data
    plt.figure(figsize=(10, 10))
    for i, (image, label) in enumerate(ds_train.take(9)):
        equalized_image = equalize_histogram(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(equalized_image, cmap='gray')
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

    rgb1 = (255/255, 109/255, 196/255)
    rgb2 = (214/255, 21/255, 150/255)
    rgb3 = (0,6/255/100,48/100)
    rgb4 = (0,167/255, 1)
    def plot_hist(hist):
        plt.plot(hist.history["accuracy"], color = rgb2)
        #does not work:
        plt.plot(hist.history["val_accuracy"], color = rgb1)
        plt.title("CNN Accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.ylim(0,1)
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()
        
    plot_hist(hist)
    
    return(hist, ds_test, model)

def predictY(ds_test,model):
    
    X_test_list = []
    y_test_list = []

    for batch in ds_test:
        X_batch, y_batch = batch
        X_test_list.append(X_batch.numpy())
        y_test_list.append(y_batch.numpy())

    # Concatenate all batches to form the complete test sets
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    y_pred = model.predict(X_test)

    return(y_test, y_pred)

def plot_roc_curve(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test[:, 0], y_pred[:, 0])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Class 0 - CNN')
    plt.legend(loc="lower right")
    plt.show()
    
    

#tests
filename_random = './randomcutouts2/41/*.fits'
filename_SL = './selected_objects_4_stefan/*.fits'

IMG_SIZE = 224
visual_test=visual_test(filename_SL, filename_random, IMG_SIZE)


#perform_ML(filename_SL, filename_random, IMG_SIZE)
hist,ds_test,model = perform_ML(filename_SL, filename_random, IMG_SIZE)

rgb1 = (255/255, 109/255, 196/255)
rgb2 = (147/255, 14/255, 103/255)
rgb3 = (0,6/255/100,48/100)
rgb4 = (0,167/255, 1)
def plot_hist(hist):
        plt.plot(hist.history["accuracy"], color = rgb2)
        plt.plot(hist.history["val_accuracy"], color = 'orange')
        plt.title("CNN Accuracy", fontsize=18)
        plt.ylabel("Accuracy", fontsize=14)
        plt.xlabel("Epoch", fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        plt.ylim(0.5,1)
        plt.legend(["Train", "Test"], loc='lower right', fontsize=14)
        plt.savefig('plot.png')
        plt.show()
        
plot_hist(hist)

#------------------------ AUC curve -------------------------
# Compute the ROC curve and AUC for class 0
y_test,y_pred = predictY(ds_test,model)
roc_display = plot_roc_curve(y_test,y_pred)
roc_display.ax_.set_title('ROC curve CNN')
roc_display.ax_.set_xlabel('False Positive Rate', fontsize=16)  # Increase fontsize as needed
roc_display.ax_.set_ylabel('True Positive Rate', fontsize=16)
plt.show()

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")

#-------------------- Save Results ----------------------------

tosave = np.array(['Accuracy ', hist.history["val_accuracy"][49], 'Runtime ', elapsed_time])
                  # , 'y_test ', y_test, 'y_pred', y_pred])

with open('MLacc_runtime.txt', 'w') as f:  # 'f' is defined within this block
    for item in tosave:
        f.write(f"{item}\n")

with open('MLy_test.txt', 'w') as f:  # 'f' is defined within this block
    for item in y_test:
        f.write(f"{item}\n")

with open('MLy_pred.txt', 'w') as f:  # 'f' is defined within this block
    for item in y_pred:
        f.write(f"{item}\n")
    