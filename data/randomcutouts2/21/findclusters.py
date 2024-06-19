#load packages
from scipy.cluster.hierarchy import fclusterdata
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
        print(data[i,:,:])
    
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
    
    files = glob.glob(filename)
    if not files:
        print('no files found')

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



#example our data

filename_random = '*.fits'
data, filename = get_data_from_file(filename_random)

#convert to rgb
data = grayscale_to_rgb(data)
#normalize and make positive
data = normalize_data(make_data_positive(data))

trial = data[290,:,:,0]
plt.imshow(trial)
plt.title(f'{filename[290]}')
plt.show()

c=fclusterdata(trial,t=0.8)#max = 1
print(c)

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

#positions of this galaxy
ra=249.9467189	
dec=60.1651242

with fits.open(filename[290]) as hdul:
    header = hdul[0].header
    print(header)

# Create WCS object from FITS header
wcs = WCS(header)

# Create SkyCoord object
sky_coord = SkyCoord(ra=ra, dec=dec, unit='deg')

# Convert RA, Dec to pixel coordinates
x_pixel, y_pixel = wcs.world_to_pixel(sky_coord)

print("Pixel coordinates (x, y):", x_pixel, y_pixel)


