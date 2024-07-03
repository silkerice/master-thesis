
#-------------------------------------------------------------------------------

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
from astropy.io import fits

#----------------------------------------------------------------------------------------------

def make_data_positive(data):
    #data can be list or array
    #convert to np.arrays
    data= np.array(data)
    for i in range(len(data[:])):
        minimum= np.min(data[i])
        eps = 0.0001
        data[i]=data[i]+abs(minimum)+eps
    
    #set all negative pixels to zero
    #data[data<0] =0
    return(data)
            
def normalize_data(data):
    #data can be list or array
    #convert to np.arrays
    #data = np.array(data)
    #normalize the data
    for i in range(len(data[:] )-1):
        data[i] = data[i]/np.max(data[i])
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

filename_SL = './Lens_simulations/*.fits'
#filename_SL = '.flat_theta_e_mocks_90k/*.fits'
IMG_SIZE = 224
hdu = 0
keys = ['NAXIS', 'FILTER']
batch_size = 64
NUM_CLASSES = 2

#filename is a string, stronglens is 1 or 0
data_SL, fitsNames_SL = get_data_from_file(filename_SL, hdu=hdu, keys=keys) #, hdu, keys

#---------------------------------------------------------------------------------------
#adding noise along the edges

plt.imshow(data_SL[0], cmap='gray')
plt.title('Original simulation')
plt.show()

#make histogram
arr = data_SL[0]
flat = arr.flatten()

negatives = flat[flat<0]
positives = -negatives
noise_pix = np.concatenate((positives, negatives))

mu = np.mean(noise_pix)
sigma = np.std(noise_pix)

image = arr

new_size = 66
# Create an empty array for the new image
new_image = np.zeros((new_size, new_size))

# Compute the starting index to place the original image in the center
start_idx = (new_size - 44) // 2

# Place the original image in the center of the new image
new_image[start_idx:start_idx + 44, start_idx:start_idx + 44] = image

# Generate Gaussian noise
noise = np.random.normal(mu, sigma, (new_size, new_size))

# Add Gaussian noise to the new image where the original image does not exist
mask = np.zeros((new_size, new_size))
mask[start_idx:start_idx + 44, start_idx:start_idx + 44] = 1
new_image = new_image * mask + noise * (1 - mask)

# Plot the new image
#plt.imshow(new_image, cmap='gray')
#plt.title('Extended Image with Gaussian Noise')
#plt.show()

#-------------------------------------------------------------------------------------
#automate the process on top

def add_noise(arr, new_size):
    size = 44
    
    #make noise
    flat = arr.flatten()
    negatives = flat[flat<0]
    positives = -negatives
    noise_pix = np.concatenate((positives, negatives))
    
    #find parameters gaussian distribution
    mu = np.mean(noise_pix)
    sigma = np.std(noise_pix)

    image = arr

    # Create an empty array for the new image
    new_image = np.zeros((new_size, new_size))

    # Compute the starting index to place the original image in the center
    start_idx = (new_size - 44) // 2

    # Place the original image in the center of the new image
    new_image[start_idx:start_idx + 44, start_idx:start_idx + 44] = image

    # Generate Gaussian noise
    noise = np.random.normal(mu, sigma, (new_size, new_size))

    # Add Gaussian noise to the new image where the original image does not exist
    mask = np.zeros((new_size, new_size))
    mask[start_idx:start_idx + 44, start_idx:start_idx + 44] = 1
    new_image = new_image * mask + noise * (1 - mask)
    
    return(new_image)


#--------------------------------------------------------------------------------------

new_size = 66

for i in range(len(data_SL)):
    arr = data_SL[i]
    data_SL[i] = add_noise(arr, new_size)


#normalize and make positive
data_SL = normalize_data(make_data_positive(data_SL))

#convert to rgb
data_SL = grayscale_to_rgb(data_SL)

#plot

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3,3, figsize=(10, 10))

# Plot each image in the grid
for i, ax in enumerate(axes.flat):
    if i < len(data_SL):
        img = data_SL[i+90]
        img_norm = (img - img.min()) / (img.max() - img.min())  # Normalize image
        ax.imshow(np.log1p(img_norm), cmap='gray')  # Use grayscale colormap
        ax.set_title(f" Simulation {i+90}",fontsize=18)
    ax.axis('off')  # Turn off axis
    
plt.tight_layout()  # Adjust layout
fig.suptitle('example simulations',y=1.05, fontsize = 20)
plt.show()

#---------------------------------------------------------------------------------------
head1 = fits.getheader('Lens_simulations/106.fits')



