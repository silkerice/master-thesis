'''
I made this python file to test why the simulations look so weird
'''


#import packages
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits
import glob
from astropy.io import fits as pyfits


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

#---------------------------------------------------------------------------------------

filename_SL = '*.fits'
IMG_SIZE = 224
hdu = 0
keys = ['NAXIS', 'FILTER']
batch_size = 64
NUM_CLASSES = 2

#filename is a string, stronglens is 1 or 0
data_SL, fitsNames_SL = get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys

#normalize and make positive
data_SL = normalize_data(make_data_positive(data_SL))

#convert to rgb
data_SL = grayscale_to_rgb(data_SL)
#np.random.shuffle(data_SL)
#data_SL = data_SL[:,31:97,31:97,:]

#plot

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(4,4, figsize=(10, 10))

# Plot each image in the grid
for i, ax in enumerate(axes.flat):
    if i < len(data_SL):
        img = data_SL[i]
        img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
        ax.imshow(np.log1p(img_norm), cmap='gray')  # Use grayscale colormap
        #ax.set_title(f" Simulation {i}",fontsize=18)
    ax.axis('off')  # Turn off axis
    
plt.tight_layout()  # Adjust layout
fig.suptitle('example clusters',y=1.05, fontsize = 20)
plt.show()

array1head = fits.getheader('/Users/silke/Documents/masterthesis/finding_clusters/clustersdata_examples/CFIS.056.241.r.0__211_338_5457_5584.fits')
