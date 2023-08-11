#import packages
from astropy import constants as const
from astropy import units as u
import csv
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import os
from astropy.io import fits

#test

hdul = fits.open('CFIS.270.299\WDR3_candID2270299000001_r.fits')
image = hdul[0].data
hdul = fits.open('CFIS.270.299\WDR3_candID2270299000001_r.psf.fits')
psf = hdul[0].data
hdul = fits.open('CFIS.270.299\WDR3_candID2270299000001_r.rms.fits')
rms = hdul[0].data

plt.imshow(image)
plt.title('image')
plt.show()

plt.imshow(np.log(psf))
plt.title('psf')
plt.show()

plt.imshow(image-rms)
plt.title('image-rms')
plt.show()

rms1 = rms
rms1[:,0:20]=rms[:,46:66]
image1 = np.where(image>0, image,rms1 )
plt.imshow(image1)
plt.title('replaced with noise')
plt.show()


#print(Angle(Angle('02h12m19s')).deg, ' ', Angle(Angle('30d33m10s')).deg)