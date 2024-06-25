
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
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve

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
#----------------------------------------------------------------------------------------------
filename_SL ='./Lens_simulations/*.fits'
filename_random = './randomcutouts2/41/*.fits'
filename_SL_real = './selected_objects_4_stefan/*.fits'
IMG_SIZE = 224
hdu = 0
keys = ['NAXIS', 'FILTER']
batch_size = 64
NUM_CLASSES = 2

#----------------------------------------------------------------------------------------------

# filename is a string, stronglens is 1 or 0
data_SL, fitsNames_SL = get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys
data_random, fitsNames_random = get_data_from_file(filename_random, hdu=hdu, keys=keys)
data_SL_real, fitsNames_SL_real = get_data_from_file(filename_SL_real, hdu=hdu, keys=keys) 

for i in range(len(data_random)):
    arr = data_random[i]
    data_random[i] = arr[10:54, 10:54]
    
for i in range(len(data_SL_real)):
    arr2 = data_SL_real[i]
    data_SL_real[i] = arr2[10:54, 10:54]
        
# Split data_random into two parts: one for training and one for testing
split_index = int(0.8 * len(data_random))
data_random_train = data_random[:split_index]
fitsNames_random_train = fitsNames_random[:split_index]

data_random_test = data_random[split_index:]
fitsNames_random_test = fitsNames_random[split_index:]

# Combine data for training: data_SL + data_random_train
data_train = np.concatenate((np.array(data_SL), np.array(data_random_train))) 
fitsNames_train = fitsNames_SL + fitsNames_random_train

# Normalize and make positive
data_train = normalize_data(make_data_positive(data_train))

# Convert to rgb
data_train = grayscale_to_rgb(data_train)

# Create labels for training
n_train = len(fitsNames_train)
nsl_train = len(fitsNames_SL)
labels_train = np.concatenate((np.ones(nsl_train), np.zeros(n_train - nsl_train)))

# Create bunch object for training
bunchobject_train = Bunch(images=data_train, targets=labels_train)

# Shuffle training data
combined_train_data = list(zip(bunchobject_train.images, bunchobject_train.targets))
np.random.shuffle(combined_train_data)
shuffled_train_images, shuffled_train_targets = zip(*combined_train_data)
shuffled_train_data = Bunch(images=np.array(shuffled_train_images), targets=np.array(shuffled_train_targets))

# Normalize the test data and make positive
data_SL_real = normalize_data(make_data_positive(data_SL_real))
data_random_test = normalize_data(make_data_positive(data_random_test))

# Convert test data to rgb
data_SL_real = grayscale_to_rgb(data_SL_real)
data_random_test = grayscale_to_rgb(data_random_test)

# Combine data for testing: data_SL_real + data_random_test
data_test = np.concatenate((np.array(data_SL_real), np.array(data_random_test)))
fitsNames_test = fitsNames_SL_real + fitsNames_random_test

# Create labels for test data
nsl_test = len(fitsNames_SL_real)
n_test = len(fitsNames_test)
labels_test = np.concatenate((np.ones(nsl_test), np.zeros(n_test - nsl_test)))

# Create bunch object for test data
bunchobject_test = Bunch(images=data_test, targets=labels_test)

# Normalize pixel values to be between 0 and 1
X_train = shuffled_train_data.images / 255.0
X_test = bunchobject_test.images / 255.0

# Rescale the images to match the input shape of the model
X_train_rescaled = tf.image.resize(X_train, (IMG_SIZE, IMG_SIZE))
X_test_rescaled = tf.image.resize(X_test, (IMG_SIZE, IMG_SIZE))

print("Shape of X_train_rescaled:", X_train_rescaled.shape)
print("Shape of X_test_rescaled:", X_test_rescaled.shape)

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

# Now, you can use X_train_flattened and X_test_flattened as inputs to your scikit-learn model

# Let's use a simple Support Vector Machine (SVM) as an example
svm_model = svm.SVC(probability=True)

# use naief bayes
# svm_model = GaussianNB()

# use random forest
# svm_model = RandomForestClassifier()

# logistic regression
# svm_model= LogisticRegression()
svm_model.fit(X_train_flattened, shuffled_train_data.targets)

# Evaluate the model
accuracy = svm_model.score(X_test_flattened, bunchobject_test.targets)
print("Accuracy:", accuracy)

#----------------------------------------------------------------------------------------------
y_pred = svm_model.predict(X_test_flattened)

# Generate the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(bunchobject_test.targets, y_pred)
disp.figure_.suptitle("Confusion Matrix - Real data")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

# The ground truth and predicted lists
y_true = []
y_pred = []
cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths and predictions to the lists
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

y_test= bunchobject_test.targets
# Predict the value of the digit on the test subset

auc_value = roc_auc_score(y_true, y_pred)

print("AUC:", auc_value)

y_scores = svm_model.predict_proba(X_test_flattened)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_scores)

roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='magenta', lw=2, label='ROC curve (area = 0.827)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC curve')
plt.legend(loc="lower right")
plt.show()

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

diff = [x-y for x, y in zip(y_true, y_pred)]
false_pos_index  = [i for i, x in enumerate(diff) if x == -1]
false_pos = X_test[false_pos_index]

#plot
# Get the first 9 images
images = false_pos[:9]

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

# Plot each image in the grid
for i, ax in enumerate(axes.flat):
    if i < len(false_pos):
        img = false_pos[i]
        img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
        ax.imshow(img_norm, cmap='gray')  # Use grayscale colormap
        ax.set_title(f" Rank = {i+false_pos_index[0]}",fontsize=18)
    #ax.axis('off')  # Turn off axis
    
plt.tight_layout()  # Adjust layout
fig.suptitle('False positive examples',y=1.05, fontsize = 20)
plt.show()

#----------------------------------------------------------------------------------------------
#find false negatives

diff = [x-y for x, y in zip(y_pred, y_true)]
false_neg_index  = [i for i, x in enumerate(diff) if x == -1]
false_neg = X_test[false_neg_index]

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
        ax.set_title(f" Rank = {i+false_neg_index[0]}",fontsize=18)
    #ax.axis('off')  # Turn off axis
    
plt.tight_layout()  # Adjust layout
fig.suptitle('False negative examples',y=1.05, fontsize = 20)
plt.show()


#----------------------------------------------------------------------------------------------

def fpr(fp, tn):
    fpr = fp/(tn+fp)
    return(fpr)

fp = len(false_pos)
tn = y_true.count(0)
fpr = fpr(fp,tn)

print('False positive rate: ', fpr)
