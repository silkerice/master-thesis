
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
import cv2

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

# Function to apply histogram equalization
def equalize_histogram(image):
    image_np = image
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


def create_class_feature(class_label, num_classes):
    class_feature = tf.one_hot(class_label, depth=num_classes)
    return class_feature

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
print('non sl length', n-nsl)
print('SL length',nsl)
#labels = create_labels(n, fitsNames)
labels = np.concatenate((np.ones(nsl), np.zeros(n-nsl)))
#labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

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
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(shuffled_data.images, shuffled_data.targets, test_size=0.2, random_state=42)

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

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
    #layers.Dense(128, activation='relu'),
    #layers.Dense(NUM_CLASSES)
])

#----------------------------------------------------------------------------------------------

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

#add feature training 
classes_train = np.random.randint(0, 5, size=len(y_train))
# Check the shapes of original_array and classes_train
print("Original train shape:", X_train_flattened.shape)
print("Classes train shape:", classes_train.shape)

# Add classes_train as a new column to original_array
new_train = np.hstack((X_train_flattened, classes_train[:, np.newaxis]))

# Check the shape of the new array
print("New train array shape:", new_train.shape)

X_train_flattened=new_train

#add feature test

classes_test = np.random.randint(0, 5, size=len(y_test))

# Check the shapes of original_array and classes_train
print("Original test shape:", X_test_flattened.shape)
print("Classes test shape:", classes_test.shape)

# Add classes_train as a new column to original_array
new_test = np.hstack((X_test_flattened, classes_test[:, np.newaxis]))

# Check the shape of the new array
print("New test array shape:", new_test.shape)

X_test_flattened=new_test

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
disp.figure_.suptitle("Confusion Matrix ")
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

#print the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle(f"Confusion Matrix - AUC = {auc}, Accuracy = {accuracy}")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

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
        img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
        #img_norm = np.log1p(img_norm)
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

'''
img = false_neg[0]
img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
plt.imshow(img_norm, cmap='gray')  # Use grayscale colormap
plt.title("False negative", fontsize=16)
plt.axis('off') 
plt.show()

img = false_pos[3]
#logtransform = np.log1p(img)
img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
plt.imshow(img_norm, cmap='gray')  # Use grayscale colormap
plt.title("False positive", fontsize=16) 
plt.axis('off') 
plt.show()
'''

def fpr(fp, tn):
    fpr = fp/(tn+fp)
    return(fpr)

fp = len(false_pos)
tn = y_true.count(0)
fpr = fpr(fp,tn)

print('False positive rate: ', fpr)
