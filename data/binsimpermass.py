#import packages
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits
from sklearn.cluster import DBSCAN
import os

#simulation Elodie -> face on
array1 = fits.getdata('Lens_simulations/106.fits')
header1 = fits.getheader('Lens_simulations/106.fits')

#simulations 90k
array2 = fits.getdata('flat_theta_e_mocks_90k/2002249022576_20180726.0.fits')
header2 = fits.getheader('flat_theta_e_mocks_90k/2002249022576_20180726.0.fits')

