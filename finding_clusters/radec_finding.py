
from astropy.io import fits
from astropy.wcs import WCS

# Load the FITS file (replace 'tile_278_299.fits' with your actual file name)
filename = './randomcutouts2/41/WDR3_candID2250300006313_r.fits'
hdu = fits.open(filename)[0]

# Extract WCS information
wcs = WCS(hdu.header)

# Get the central pixel coordinates (assuming the tile is centered)
nx, ny = hdu.data.shape
center_pixel = [nx / 2, ny / 2]

# Convert pixel coordinates to RA and Dec
ra_dec = wcs.pixel_to_world(center_pixel[0], center_pixel[1])
print(f"Central RA: {ra_dec.ra.deg} degrees")
print(f"Central Dec: {ra_dec.dec.deg} degrees")

# Open the FITS file
hdul = fits.open(filename)

# Access the WCS information from the header
wcs = WCS(hdul[0].header)  # Assuming WCS info is in the primary header

# Accessing the primary header (usually the first HDU)
primary_header = hdul[0].header

# Accessing headers and data from other HDUs if present
for hdu in hdul:
    header = hdu.header
    data = hdu.data
    # Process header or data as needed

hdul.close()



