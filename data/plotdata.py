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



#---------------------------------------------------------------------------------------
head1 = fits.getheader('selected_objects_4_stefan/WDR3_candID2017253021765_r.fits')

from astropy.io import fits

def plot_fits_from_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter for FITS files
    fits_files = [f for f in files if f.lower().endswith('.fits')]

    # Select up to 9 FITS files
    files_to_plot = fits_files[30:34]#04

    # Create a 3x3 grid for plotting
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()  # Flatten the 2D array to 1D for easy indexing

    for ax, fits_file in zip(axes, files_to_plot):
        fits_path = os.path.join(directory, fits_file)
        
        # Open the FITS file and read the data
        with fits.open(fits_path) as hdul:
            # Assuming the data is in the first extension
            data = hdul[0].data
        
        # Plot the data
        ax.imshow(data, cmap='gray', origin='lower')
        ax.axis('off')  # Hide the axis

    # Hide any unused axes if there are less than 9 images
    for ax in axes[len(files_to_plot):]:
        ax.axis('off')

    plt.tight_layout()
    # Save the figure as a PDF
    plt.savefig('ex_nsl', format='pdf')
    plt.show()

# Replace 'your_directory_path' with the path to your directory
directory_path = 'randomcutouts2/11/'
plot_fits_from_directory(directory_path)
