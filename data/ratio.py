
import time

# Record the start time
start_time = time.time()

#import packages
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits
import glob
from astropy.io import fits as pyfits

#tensor packages
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model


# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

#----------------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------------
#filename_SL = './flat_theta_e_mocks_90k/*.fits'
filename_SL = './selected_objects_4_stefan/*.fits'
filename_random = './randomcutouts2/21/*.fits'
IMG_SIZE = 224
hdu = 0
keys = ['NAXIS', 'FILTER']
batch_size = 64
NUM_CLASSES = 2

#----------------------------------------------------------------------------------------------

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

#create labels
n = len(fitsNames)
nsl = len(fitsNames_SL)
#labels = create_labels(n, fitsNames)
labels = np.concatenate((np.ones(nsl), np.zeros(n-nsl)))
#labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

print(len(labels))

#----------------------------------------------------------------------------------------------

#rotate and flip stuff 
listnewimages = []
listnewlabels = []
for i in range(len(labels)):
    #check label
    if labels[i] == 1:
        y=1
    else:
        y=0
    
    for j in range(6):
        listnewlabels.append(y)
        
    #load image
    image_array = data[i,:,:,:]
    listnewimages.append(image_array)
    # Flip the image horizontally
    flipped_horizontal = np.fliplr(image_array)
    listnewimages.append(flipped_horizontal)
        
    # Flip the image vertically
    flipped_vertical = np.flipud(image_array)
    listnewimages.append(flipped_vertical)
    
    #rotated and flipped images
    for j in [1,2,3]:
        #rotated
        rot =np.rot90(image_array, j)
        listnewimages.append(rot)   
  
labels = np.array(listnewlabels)
data =np.array(listnewimages)

#plot as example
fig, axes = plt.subplots(3,3, figsize=(10, 10))
# Plot each image in the grid\=
for i, ax in enumerate(axes.flat):
    if i < len(data):
        index = i
        img = data[index]
        img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
        ax.imshow(np.log1p(img_norm), cmap='gray')  # Use grayscale colormap
    ax.axis('off')  # Turn off axis

#create bunch object
bunchobject = Bunch(images = data, targets = labels)

#shuffle data
combined_data = list(zip(bunchobject.images, bunchobject.targets))

# Shuffle the combined data
np.random.shuffle(combined_data)

# Split the shuffled data back into images and targets
shuffled_images, shuffled_targets = zip(*combined_data)

# Create a new Bunch object with shuffled data
shuffled_data = Bunch(images=np.array(shuffled_images), targets=np.array(shuffled_targets))

#----------------------------------------------------------------------------------------------
#data = data[0:2000,:,:,:]
#labels=labels[0:2000]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(shuffled_data.images, shuffled_data.targets, test_size=0.3, random_state=42)

#for the splitting
#training set
ones = np.where(y_train==1)[0]
zeros =np.where(y_train==0)[0]
print('number of ones train', len(ones))
print('number of zeros train', len(zeros))
      
lentrain = len(y_train)
newindex0 = zeros[0:len(ones)]
newindex1 = ones#[0:1000]
index=np.concatenate((newindex0,newindex1))

X_train = X_train[index]
y_train= y_train[index]

#check
ones = np.where(y_train==1)
zeros =np.where(y_train==0)
print('number of ones train', len(ones[0]))
print('number of zeros train', len(zeros[0]))

#testset
ones = np.where(y_test==1)[0]
zeros =np.where(y_test==0)[0]
print('number of ones test', len(ones))
print('number of zeros test', len(zeros))
newindex1 = ones
newindex0 = zeros[0:len(ones)]
index=np.concatenate((newindex0,newindex1))

X_test = X_test[index]
y_test= y_test[index]

#check
ones = np.where(y_test==1)
zeros =np.where(y_test==0)
print('number of ones test', len(ones[0]))
print('number of zeros test', len(zeros[0]))

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Rescale the images to match the input shape of the model
X_train_rescaled = tf.image.resize(X_train, (IMG_SIZE, IMG_SIZE))
X_test_rescaled = tf.image.resize(X_test, (IMG_SIZE, IMG_SIZE))

print("Shape of X_train_rescaled:", X_train_rescaled.shape)
print("Shape of X_test_rescaled:", X_test_rescaled.shape)

#----------------------------------------------------------------------------------------------
#data = data[0:2000,:,:,:]
# Define your TensorFlow model

model = Sequential([
    layers.Rescaling(1.0/255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    #layers.Dense(128, activation='relu'),
    #layers.Dense(NUM_CLASSES)
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Iterate through the layers of the model to find the last convolutional layer
last_conv_layer = None
for layer in model.layers[::-1]:
    if 'conv' in layer.name:
        last_conv_layer = layer
        break

# Check if the last convolutional layer is found
if last_conv_layer is None:
    raise ValueError("No convolutional layer found in the model.")
else:
    print(last_conv_layer)
    

# Define the pre-flattening part of the model
pre_flattening_model = Model(inputs=model.input, outputs=last_conv_layer.output)

# Get the output of the last convolutional layer before flattening
X_train_conv = pre_flattening_model.predict(X_train_rescaled)
X_test_conv = pre_flattening_model.predict(X_test_rescaled)

# Now, flatten the data
X_train_flattened = X_train_conv.reshape(X_train_conv.shape[0], -1)
X_test_flattened = X_test_conv.reshape(X_test_conv.shape[0], -1)


#----------------------------------------------------------------------------------------------

# Now, you can use X_train_flattened and X_test_flattened as inputs to your scikit-learn model

# Let's use a simple Support Vector Machine (SVM) as an example
svm_model = svm.SVC(probability = True)

#use naief bayes
#svm_model = GaussianNB()

#use random forest
#svm_model = RandomForestClassifier()

#logistic regression
#svm_model= LogisticRegression()
svm_model.fit(X_train_flattened, y_train)

# Evaluate the model
accuracy = svm_model.score(X_test_flattened, y_test)
print("Accuracy:", accuracy)

#----------------------------------------------------------------------------------------------
y_pred = svm_model.predict(X_test_flattened)
originalpred = y_pred

# Predict the value of the digit on the test subset
predicted = svm_model.predict(X_test_flattened)

print(f"Classification report for classifier {svm_model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")

#print the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix - Real data")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

#If the results from evaluating a classifier are stored in the form of a confusion 
#matrix and not in terms of y_true and y_pred, one can still build a 
#classification_report as follows:

# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
for gt in range(len(cm)):
    for pred in range(len(cm)):
        y_true += [gt] * cm[gt][pred]
        y_pred += [pred] * cm[gt][pred]

print(
    "Classification report rebuilt from confusion matrix:\n"
    f"{metrics.classification_report(y_true, y_pred)}\n"
)



#----------------------------------------------------------------------------------------------
#AUC calculation

auc = roc_auc_score(y_true, y_pred)

print("AUC:", auc)

#----------------------------------------------------------------------------------------------
#calibration of classifiers

#calibration curves
#plot calibration curves (also known as reliability diagrams) using predicted 
#probabilities of the test dataset.

class NaivelyCalibratedLinearSVC(LinearSVC):
    """LinearSVC with `predict_proba` method that naively scales
    `decision_function` output."""

    def fit(self, X, y):
        super().fit(X, y)
        df = self.decision_function(X)
        self.df_min_ = df.min()
        self.df_max_ = df.max()

    def predict_proba(self, X):
        """Min-max scale output of `decision_function` to [0,1]."""
        df = self.decision_function(X)
        calibrated_df = (df - self.df_min_) / (self.df_max_ - self.df_min_)
        proba_pos_class = np.clip(calibrated_df, 0, 1)
        proba_neg_class = 1 - proba_pos_class
        proba = np.c_[proba_neg_class, proba_pos_class]
        return proba

#calculate predicted possibilities
prob_pos = svm_model.predict_proba(X_test_flattened)[:, 1]

#plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=5)


plt.figure(figsize=(8, 8))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="SVM", color = 'magenta')
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability",fontsize = 14)
plt.ylabel("Fraction of positives",fontsize = 14)
plt.title("Calibration plot - SVM", fontsize = 16)
plt.legend(fontsize = 14)
plt.show()

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")

#----------------------------------------------------------------------------------------------
#find false positives

diff = [x-y for x, y in zip(y_test, originalpred)] #THIS CHANGED!!
#diff = [x-y for x, y in zip(y_true, y_pred)] 
false_pos_index  = [i for i, x in enumerate(diff) if x == -1]
false_pos = X_test[false_pos_index]
#find probability of being strong lens for each image
prob_pos_diff=prob_pos[false_pos_index]

#find rank
l = sorted(prob_pos)
rank = []
#search which index in list 
for i in range(len(prob_pos_diff)):
    index = l.index(prob_pos_diff[i])
    rank.append(index)

#plot
# Get the first 9 images
images = false_pos[:9]

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Plot each image in the grid
for i, ax in enumerate(axes.flat):
    if i < len(false_pos):
        img = false_pos[i]
        
        logtransform = np.log1p(img)
        img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
        ax.imshow(img_norm, cmap='gray')  # Use grayscale colormap
        ax.set_title(f" Prob of SL:{ np.round(prob_pos_diff[i],2)}, rank = {rank[i]}", fontsize=16)
    ax.axis('off')  # Turn off axis
    
plt.tight_layout()  # Adjust layout
cutoff = cm[0,0]+cm[0,1]
fig.suptitle(f'False positive examples, cutoff = {cutoff}',y=1.05, fontsize = 20)
plt.show()

#----------------------------------------------------------------------------------------------
#find false negatives

diff = [x-y for x, y in zip(originalpred, y_test)] #THIS CHANGED!!
#diff = [x-y for x, y in zip(y_pred, y_true)]
false_neg_index  = [i for i, x in enumerate(diff) if x == -1]
false_neg = X_test[false_neg_index]
#find probability of being strong lens for each image
prob_pos_diff=prob_pos[false_neg_index]

#find rank
l = sorted(prob_pos)
rank = []
#search which index in list 
for i in range(len(prob_pos_diff)):
    index = l.index(prob_pos_diff[i])
    rank.append(index)

#plot
# Get the first 9 images
images = false_neg[:9]

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3,3, figsize=(10, 10))

# Plot each image in the grid
for i, ax in enumerate(axes.flat):
    if i < len(false_neg):
        img = false_neg[i]
        img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
        ax.imshow(img_norm, cmap='gray')  # Use grayscale colormap
        ax.set_title(f" Prob of SL:{ np.round(prob_pos_diff[i],2)}, rank = {rank[i]}", fontsize=16)
    ax.axis('off')  # Turn off axis
    
plt.tight_layout()  # Adjust layout
cutoff = cm[0,0]+cm[0,1]
fig.suptitle(f'False negative examples, cutoff = {cutoff}',y=1.05, fontsize = 20)
plt.show()


