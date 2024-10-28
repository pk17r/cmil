
##########################################

##       ENTER PATIENT IMAGE ID         ##

patient_id = "h2114154 h&e"

##########################################


## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
from skimage import filters

import os

# Plot the image
def imshow(img, title):
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.title(title)
    plt.show(block=True)

extracted_dir = "extracted"

segmented_images_dir = "segmented"


sub_img_num =1
filename = os.path.join(extracted_dir, patient_id, patient_id + " series[0] img#" + str(sub_img_num))
print(filename + " Lumma.png")

#Image loading
img_color = cv2.imread(filename + ".png")
#imshow(img_color, 'img_color')

lumma_channel = cv2.imread(filename + " Lumma.png", cv2.IMREAD_GRAYSCALE)
#imshow(lumma_channel, "lumma_channel")

img_gray= cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
#imshow(img_gray, "img_gray")

fig, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 12))
fig.suptitle('RGB Grayscale vs YUV Lumma - Stroma-Epithelia better differentiated', fontsize = 25)
ax1, ax2, ax3 = ax_arr.ravel()
ax1.imshow(img_color)
ax1.set_title('RGB888', fontsize = 20)
ax1.set_axis_off()
ax2.imshow(img_gray)
ax2.set_title('RGB888_to_Grayscale', fontsize = 20)
ax2.set_axis_off()
ax3.imshow(lumma_channel)
ax3.set_title('YCbCr_Y(Luminance)_channel', fontsize = 20)
ax3.set_axis_off()
plt.tight_layout()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " RGB Grayscale vs YUV Lumma.png")
plt.savefig(filename)
#plt.show()
plt.close()
del fig

# smaller lumma band
lumma_band = lumma_channel[3600:3800,:]
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Band.png")
cv2.imwrite(filename, lumma_band, [cv2.IMWRITE_PNG_COMPRESSION , 0])

# Histogram
fig = plt.hist(lumma_band, bins=10) 
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('lumma_band Histogram')
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Band Histogram.png")
plt.savefig(filename)
plt.show()

# decimate image into 10 bands
lumma_band_decimated = lumma_band/25
lumma_band_decimated = (np.floor(lumma_band/25)).astype(np.uint8)
imshow(lumma_band_decimated, 'lumma_band_decimated')
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Band Decimated.png")
cv2.imwrite(filename, lumma_band_decimated * 25, [cv2.IMWRITE_PNG_COMPRESSION , 0])

print("lumma_band_decimated range = [" + str(np.min(lumma_band_decimated)) + "," + str(np.max(lumma_band_decimated)) + "]")

imshow(lumma_band_decimated == 6, 'lumma_band_decimated')

# figure to show different decimated values
fig, ax_arr = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(20, 12))
fig.suptitle('Lumma Band Decimated range=[' + str(np.min(lumma_band_decimated)) + "," + str(np.max(lumma_band_decimated)) + "]", fontsize = 25)
row = 0
col = 0
for decimated_band_i in range(1,11):
    print("decimated_band_i = " + str(decimated_band_i) + "  row = " + str(row) + "  col = " + str(col))
    if decimated_band_i < 10:
        ax_arr[row, col].set_title("value <= " + str(decimated_band_i), fontsize = 20)
    else:
        ax_arr[row, col].set_title("value <= " + str(decimated_band_i) + " (all true)", fontsize = 20)
    ax_arr[row, col].set_axis_off()
    ax_arr[row, col].imshow(lumma_band_decimated <= decimated_band_i)
    row = row +1
    if row == 5:
        row = 0
        col = 1

plt.tight_layout()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Band Decimated Values 1.png")
plt.savefig(filename)
plt.show()
plt.close()
del fig


# figure to show different decimated values
fig, ax_arr = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(20, 12))
fig.suptitle('Lumma Band Decimated range=[' + str(np.min(lumma_band_decimated)) + "," + str(np.max(lumma_band_decimated)) + "]", fontsize = 25)
row = 0
col = 0
for decimated_band_i in range(1,11):
    print("decimated_band_i = " + str(decimated_band_i) + "  row = " + str(row) + "  col = " + str(col))
    ax_arr[row, col].set_title("value == " + str(decimated_band_i), fontsize = 20)
    ax_arr[row, col].set_axis_off()
    ax_arr[row, col].imshow(lumma_band_decimated == decimated_band_i)
    row = row +1
    if row == 5:
        row = 0
        col = 1

plt.tight_layout()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Lumma Band Decimated Values 2.png")
plt.savefig(filename)
plt.show()
plt.close()
del fig


# fill in values 0-3 from nearby regions




lumma_f = lumma_band.astype(np.float64)
lumma_f = lumma_f / 255
#imshow(lumma_f, 'lumma_f')
test = (lumma_f < 0.8) & (lumma_f > 0.7)
imshow(test, 'test')
lumma_fg = filters.gaussian(lumma_f, sigma=10)#, preserve_range=True)
imshow(lumma_fg, 'lumma_fg')
imshow(lumma_f < 0.3, 'black_holes')

# gaussian filter
#gaussian_smooth = filters.gaussian(lumma_channel, sigma=10)#, preserve_range=True)
#imshow(gaussian_smooth, 'gaussian_smooth')

# Compute a mask
mskg = gaussian_smooth < 0.7
imshow(mskg, 'mskg')
msk1 = morphology.remove_small_objects(mskg, 500)
#imshow(msk1, 'msk1')
msk2 = morphology.remove_small_holes(msk1, 4000)
#imshow(msk2, 'msk2')

# dilation
dilated_lumma_mask = morphology.dilation(msk2, morphology.square(25))
#imshow(dilated_lumma_mask, 'dilated_mask')
remove_small_holes = morphology.remove_small_holes(dilated_lumma_mask, 5000)
mask = morphology.remove_small_objects(remove_small_holes, 5000)

#mask = morphology.opening(dilated_mask, morphology.disk(10))
#imshow(mask, 'mask')

# show mask
fig, ax_arr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 12))
ax1, ax2, ax3, ax4 = ax_arr.ravel()
ax1.imshow(img_color)
ax1.set_title('img_color', fontsize = 25)
ax1.set_axis_off()
ax2.imshow(lumma_channel)
ax2.set_title('lumma_channel', fontsize = 25)
ax2.set_axis_off()
ax3.imshow(mask)
ax3.set_title('mask', fontsize = 25)
ax3.set_axis_off()
ax4.imshow(segmentation.mark_boundaries(img_color, mask))
ax4.contour(mask, colors='red', linewidths=3)
ax4.set_title('img tumor', fontsize = 25)
ax4.set_axis_off()
plt.tight_layout()
#plt.show()
filename = os.path.join(segmented_images_dir, patient_id + " img#" + str(sub_img_num) + " Tumor.png")
plt.savefig(filename)
print("'" + filename + "' saved!")
plt.close()

