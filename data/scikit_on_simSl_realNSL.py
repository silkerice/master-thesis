#scikit learn on actual data and simulated data

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from astropy.io import fits
import glob
from astropy.io import fits as pyfits

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
from sklearn.metrics import roc_auc_score

from skimage.transform import resize
#-------------------------------------------------------------------------------------

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
        arr = data[i,:,:]
        arr[arr== 255] =0
        data[i,:,:] = arr[:,:]/np.max(arr[:,:])
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

#------------------------------------------------------------------------------------
filename_SL='./flat_theta_e_mocks_90k/*.fits'
filename_random='./randomcutouts2/*.fits'
IMG_SIZE = 224
hdu=0
keys =['NAXIS', 'FILTER']

#------------------------------------------------------------------------------------
#prepare data

#read data
'''
data_SL, fitsNames_SL = get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys
data_random, fitsNames_random = get_data_from_file(filename_random, hdu=hdu, keys=keys)

#put together
#transform to arrays
data_random=np.array(data_random)
data_SLnew = np.zeros((460,66,66))
for i in range(len(data_SL)):
    array = np.array(data_SL[i])
    if (np.shape(array) == (480, 640, 4)):
        array = array[:,:,0]
    array = resize(array, (66,66))
    data_SLnew[i,:,:] = array
print(np.shape(data_SLnew))
data_SL = data_SLnew

data = np.concatenate((data_SL, data_random)) 
fitsNames = fitsNames_SL + fitsNames_random

#normalize and make positive
data = normalize_data(make_data_positive(data))
'''

#filename is a string, stronglens is 1 or 0
data_SL, fitsNames_SL = get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys
data_random, fitsNames_random = get_data_from_file(filename_random, hdu=hdu, keys=keys)
#put together
data = np.concatenate((np.array(data_SL), np.array(data_random ))) 
fitsNames = fitsNames_SL + fitsNames_random
print('concetanate successful')
#normalize and make positive
data = normalize_data(make_data_positive(data))


#create labels
n = len(fitsNames)
nsl = len(fitsNames_SL)
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

# flatten the images
n_samples = len(shuffled_data.images)
data = shuffled_data.images.reshape((n_samples, -1))

#------------------------------------------------------------------------------------
#perform ML

# Create a classifier: a support vector classifier
#clf = svm.SVC()
#clf = GaussianNB()
clf = RandomForestClassifier()
#clf=LogisticRegression()

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, shuffled_data.targets, test_size=0.5, shuffle=True
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print(f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")

#print the confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
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

#---------------------------------------------------------------------------------
#calibration of classifiers - SYNTHETIC!!

#dataset
#synthetic binary classification dataset with 100,000 samples and 20 features. 
#Of the 20 features, only 2 are informative, 2 are redundant (random combinations
# of the informative features) and the remaining 16 are uninformative 
#(random numbers). Of the 100,000 samples, 100 will be used for model fitting 
#and the remaining for testing.

X, y = make_classification(
    n_samples=100_000, n_features=20, n_informative=2, n_redundant=2, random_state=42
)

train_samples = 100  # Samples used for training the models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=False,
    test_size=100_000 - train_samples,
)

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

# Create classifiers
lr = LogisticRegression()
gnb = GaussianNB()
svc = NaivelyCalibratedLinearSVC(C=1.0, dual="auto")
rfc = RandomForestClassifier()

clf_list = [
    (lr, "Logistic"),
    (gnb, "Naive Bayes"),
    (svc, "SVC"),
    (rfc, "Random forest"),
]

fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}
markers = ["^", "v", "s", "o"]
for i, (clf, name) in enumerate(clf_list):
    clf.fit(X_train, y_train)
    display = CalibrationDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        n_bins=10,
        name=name,
        ax=ax_calibration_curve,
        color=colors(i),
        marker=markers[i],
    )
    calibration_displays[name] = display

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")

# Add histogram
grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
for i, (_, name) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=10,
        label=name,
        color=colors(i),
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

plt.tight_layout()
plt.show()


#---------------------------------------------------------------------------------
#compute AUC 

auc = roc_auc_score(y_true, y_pred)

print("AUC:", auc)







