#import packages
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits
from sklearn.cluster import DBSCAN
import os
'''
array1 = fits.getdata('Lens_simulations/106.fits')

array1 = array1 + np.abs(np.min(array1)) +0.0001
array1 = array1/np.max(array1)

array2 = fits.getdata('Lens_simulations/1050.fits')
array2 = array2 + np.abs(np.min(array2)) +0.0001
array2 = array2/np.max(array2)

array3 = fits.getdata('Lens_simulations/1051.fits')
array3 = array3 + np.abs(np.min(array3)) +0.0001
array3 = array3/np.max(array3)

array4 = fits.getdata('Lens_simulations/110677.fits')
array4 = array4 + np.abs(np.min(array4)) +0.0001
array4 = array4/np.max(array2)

data = np.array([array1, array2, array3, array4])

# Visualizing the data
plt.figure(figsize=(10, 10))
for i, image in enumerate(data):
    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(image, cmap='gray')
    plt.axis("off")

plt.show()
'''
'''
headers = []
fitsNames = []
data = []

hdu = 0

import glob

filename = './flat_theta_e_mocks_90k/*.fits'

for fitsName in glob.glob(filename):
    # opening the file is unnecessary. just pull the (right) header
    header = fits.getheader(fitsName, hdu)
    headers.append(header)
    
    #save data
    dat = fits.getdata(fitsName)
    data.append(dat)
    
    #save filename
    fitsNames.append(fitsName)
'''



# List of image file paths
image_files = [
    'correctSL1.png', 'correctSL2.png', 'correctSL3.png',
    'correctNSL1.png', 'correctNSL2.png', 
    'SLasNSL.png', 'SLasNSL2.png', 'NSLasSL2.png', 'NSLasSL.png'
]

# List of subtitles for each image 
subtitles = [
    'SL classified as SL', 'SL classified as SL', 'SL classified as SL',
    'NSL classified as NSL', 'NSL classified as NSL', 'SL misclassified as NSL',
    'SL misclassified as NSL', 'NSL misclassified as SL', 'NSL misclassified as SL'
]

import matplotlib.image as mpimg

# Create a figure and a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 10))

# Loop over the image files and axes to plot each image
for ax, img_path, subtitle in zip(axs.flat, image_files, subtitles):
    # Load the image
    img = mpimg.imread(img_path)
    # Display the image
    ax.imshow(img)
    # Set subtitle for each subplot
    ax.set_title(subtitle, fontsize=16)
    # Remove axis labels
    ax.axis('off')

plt.suptitle('LTR classification examples', fontsize = 18)
# Adjust layout
plt.savefig('plot.png')
plt.tight_layout()

# Show the plot
plt.show()

#---------------------------------------------------------------------------------------
head1 = fits.getheader('selected_objects_4_stefan/WDR3_candID2017253021765_r.fits')
