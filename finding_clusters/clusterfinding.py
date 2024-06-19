
#import packages
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from astropy.io import fits
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata

df= pd.read_table('all_gal_1deg2.txt',sep = ',')

x_pixels=np.array(df['X_IMAGE'])
y_pixels=np.array(df['Y_IMAGE'])
ra=np.array(df['RA'])
dec=np.array(df['Dec'])
ids = np.array(df['ID'])

X = np.vstack((x_pixels,y_pixels)).T

pix_scale = 5.160234650248e-05 *3600 #arcsec
sep=1 #trt 1 through 5
search_size=sep/pix_scale

Y=fclusterdata(X[50000:70000,:], t=search_size, criterion='distance')

#find maxim
num_cluster = np.unique(Y)

num_gals = np.zeros(len(num_cluster))

for i in range(len(num_cluster)):
    num = num_cluster[i]
    num_gals[i]= len(np.where(Y == num)[0])

num_cutouts = 200

num_gal_s, num_cluster_s = zip(*sorted(zip(num_gals,num_cluster), reverse=True))

topten = num_cluster_s[0:num_cutouts]


gal_ids=[]

for i in range(len(topten)):
    indices = np.where(Y == topten[i])[0]
    gal_ids.append(ids[indices[0]])
    
# Convert each item to a string using list comprehension
gal_ids_str = [str(item) for item in gal_ids]
    
#write out to txt file, get cutouts, fiddle with t
# Open a file in write mode
with open("gall_ids.txt", "w") as file:
    # Iterate over the list and write each item to the file
    for item in gal_ids_str:
        file.write(item+ ',')  # Add a newline after each item
        
#_________________________find positions________________________________________

ids_list = []
for i in range(len(gal_ids)):
   ids_list.append(np.where(ids == gal_ids[i])[0])

ra_list = []
dec_list=[]

for i in range(len(ids_list)):
    ra_list.append(ra[(ids_list[i][0])])
    dec_list.append(dec[(ids_list[i][0])])

# Convert each item to a string using list comprehension
ra_list_str = [str(item) for item in ra_list]
dec_list_str = [str(item) for item in dec_list]

with open("gal_pos.txt", "w") as file:
    for ra, dec in zip(ra_list_str, dec_list_str):
        file.write(f"{ra} {dec}\n")
        
        
        
array1 = fits.getdata('WDR3_candID2250300016449_r.fits')
plt.imshow(array1)


