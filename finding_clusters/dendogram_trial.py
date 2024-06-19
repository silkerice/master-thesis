
#import packages
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits
from sklearn.cluster import DBSCAN


array1 = fits.getdata('WDR3_candID2250300015869_r.fits')

array1 = array1 + np.abs(np.min(array1)) +0.0001
array1 = array1/np.max(array1)

#plot
plt.imshow(array1)
plt.title('array1')
plt.show()

array2 = fits.getdata('WDR3_candID2250300015184_r.fits')
array2 = array2 + np.abs(np.min(array2)) +0.0001
array2 = array2/np.max(array2)

#plot
plt.imshow(array2)
plt.title('array2')
plt.show()


# Flatten the image
flattened_image = array1.flatten()

# Reshape the flattened array to 2D
reshaped_image = flattened_image.reshape(-1, 1)

db=DBSCAN(eps=2, min_samples=10).fit(reshaped_image)

labels1 = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_1 = len(set(labels1)) #- (1 if -1 in labels1 else 0)

print("Estimated number of clusters arr1: %d" % n_clusters_1)


# Flatten the image
flattened_image = array2.flatten()

# Reshape the flattened array to 2D
reshaped_image = flattened_image.reshape(-1, 1)

db=DBSCAN(eps=2, min_samples=10).fit(reshaped_image)

labels2 = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_2 = len(set(labels2)) - (1 if -1 in labels2 else 0)

print("Estimated number of clusters arr2: %d" % n_clusters_2)