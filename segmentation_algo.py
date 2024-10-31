
##########################################

##       SEGMENTATION ALGORITHM         ##

show_images = 1
save_intermediate_images = 0
input_dir = "data/sheffield_h&e"
#output_dir = "extracted/sheffield_h&e"
output_dir = "workingdir/segmented"
run_over_all_images = 0
image_name = "h2114186 h&e_ROI_3"

# visualizations
#output_visualization_dir = "extracted/sheffield_h&e/visualization"
output_visualization_dir = "workingdir/segmented"
save_rgb_stroma_epithelia_comparison = 1
save_bins_representation = 1


##########################################


## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation
from skimage import filters
import scipy.ndimage as ndimage

# Plot the image
def imshow(img, title):
    global show_images
    if show_images == 0:
        return
    
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.title(title)
    plt.show(block=True)

def pixels_within_distance(mask, distance):
    """Finds pixels within a specified distance from a 2D mask."""
    # Create a distance map from the mask
    distance_map = ndimage.distance_transform_edt(np.logical_not(mask))
    # Find pixels within the specified distance
    pixels_within = distance_map <= distance
    return pixels_within



if not os.path.isdir(input_dir):
    print("input_dir: '" + input_dir + "' directory does not exist! Exiting...")
    exit()

if not os.path.isdir(output_dir):
    print("output_dir: '" + output_dir + "' directory does not exist! Exiting...")
    exit()

if save_rgb_stroma_epithelia_comparison or save_bins_representation:
    if not os.path.isdir(output_visualization_dir):
        print("output_visualization_dir: '" + output_visualization_dir + "' directory does not exist! Exiting...")
        exit()
    

# Files and Folders in Input Dir
files = os.listdir(input_dir)
# Filtering only the files.
files = [f for f in files if os.path.isfile(input_dir+'/'+f)]

for f in files:
    print("\n*********** CURRENT FILE: " + f)
    x = f.split(".", 1)
    if x[1] != "tif":
        print(x)
        print("Unexpected Filename - more dots than anticipated. Exiting...")
        exit()
    
    if run_over_all_images:
        image_name = x[0]
    
    input_filepath = os.path.join(input_dir, image_name)
    print(input_filepath + ".tif")
    
    #Image loading
    tif = tf.TiffFile(input_filepath + ".tif")
    img_rgb = tif.series[0].asarray()
    #img_rgb = cv2.imread(filename + ".tif")
    #imshow(img_rgb, 'img_rgb')
    if show_images:
        fig, ax_arr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 12))
        fig.suptitle('RGB - R - G - B', fontsize = 25)
        ax1, ax2, ax3, ax4 = ax_arr.ravel()
        ax1.imshow(img_rgb)
        ax1.set_title('RGB888', fontsize = 20)
        ax1.set_axis_off()
        ax2.imshow(img_rgb[:,:,0])
        ax2.set_title('Red Channel', fontsize = 20)
        ax2.set_axis_off()
        ax3.imshow(img_rgb[:,:,1])
        ax3.set_title('Green Channel', fontsize = 20)
        ax3.set_axis_off()
        ax4.imshow(img_rgb[:,:,2])
        ax4.set_title('Blue Channel', fontsize = 20)
        ax4.set_axis_off()
        plt.tight_layout()
        if save_bins_representation:
            filename = os.path.join(output_visualization_dir, image_name + " RGB Channels.png")
            plt.savefig(filename)
            print(filename + " saved")
        
        plt.show()
        plt.close()
        del fig
    
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    #imshow(img_ycrcb, 'img_ycrcb')
    if show_images:
        fig, ax_arr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 12))
        fig.suptitle('YCbCr - Y - Cb - Cr', fontsize = 25)
        ax1, ax2, ax3, ax4 = ax_arr.ravel()
        ax1.imshow(img_ycrcb)
        ax1.set_title('YCbCr', fontsize = 20)
        ax1.set_axis_off()
        ax2.imshow(img_ycrcb[:,:,0])
        ax2.set_title('Y Channel', fontsize = 20)
        ax2.set_axis_off()
        ax3.imshow(img_ycrcb[:,:,1])
        ax3.set_title('Cb Channel', fontsize = 20)
        ax3.set_axis_off()
        ax4.imshow(img_ycrcb[:,:,2])
        ax4.set_title('Cr Channel', fontsize = 20)
        ax4.set_axis_off()
        plt.tight_layout()
        if save_bins_representation:
            filename = os.path.join(output_visualization_dir, image_name + " YCbCr Channels.png")
            plt.savefig(filename)
            print(filename + " saved")
        
        plt.show()
        plt.close()
        del fig
    
    img_lumma = img_ycrcb[:,:,0]
    imshow(img_lumma, "img_lumma")
    
    img_Cr = img_ycrcb[:,:,2]
    imshow(img_lumma, "img_Cr")
    
    #img_gray= cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    #imshow(img_gray, "img_gray")
    
    #fig, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 12))
    #fig.suptitle('Grayscale vs YCbCr Lumma - Stroma-Epithelia better differentiated', fontsize = 25)
    #ax1, ax2, ax3 = ax_arr.ravel()
    #ax1.imshow(img_rgb)
    #ax1.set_title('RGB888', fontsize = 20)
    #ax1.set_axis_off()
    #ax2.imshow(img_gray)
    #ax2.set_title('RGB888_to_Grayscale', fontsize = 20)
    #ax2.set_axis_off()
    #ax3.imshow(img_lumma)
    #ax3.set_title('YCbCr_Y(Luminance)', fontsize = 20)
    #ax3.set_axis_off()
    #plt.tight_layout()
    #filename = os.path.join(output_visualization_dir, image_name + " RGB Grayscale vs YUV Lumma.png")
    #plt.savefig(filename)
    #print(filename + " saved")
    ##plt.show()
    #plt.close()
    #del fig
    
    
    # define binning in lumma image
    lumma_bins_n = 20
    divisor = (np.floor(255 / lumma_bins_n).astype(np.uint8))
    
    # decimate lumma image into lumma_bins_n
    lumma_binned = (np.floor(img_lumma/divisor)).astype(np.uint8)
    #filename = os.path.join(output_visualization_dir, image_name + " Lumma " + str(lumma_bins_n) + " Bins.png")
    #cv2.imwrite(filename, lumma_binned * divisor, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #imshow(lumma_binned, "lumma_binned " + str(lumma_bins_n))
    #print(filename + " saved")
    
    ## figure to show different lumma bins
    if save_bins_representation:
        fig, ax_arr = plt.subplots(2, int(lumma_bins_n/2) + 1, sharex=True, sharey=True, figsize=(20, 12))
        fig.suptitle(image_name + " Lumma " + str(lumma_bins_n) + " Bins Representation", fontsize = 25)
        row=0
        col=0
        for bin_i in range(0,lumma_bins_n + 2):
            ax_arr[row,col].set_title("bin " + str(bin_i), fontsize = 20)
            ax_arr[row,col].set_axis_off()
            ax_arr[row,col].imshow(lumma_binned == bin_i)
            col=col+1
            if col == int(lumma_bins_n/2) + 1:
                row = 1
                col = 0
        
        plt.tight_layout()
        filename = os.path.join(output_visualization_dir, image_name + " Lumma " + str(lumma_bins_n) + " Bins Representation.png")
        plt.savefig(filename)
        print(filename + " saved")
        if show_images:
            plt.show()
        
        plt.close()
        del fig
    
    
    # find bin with most number of pixels
    
    most_pixels_bin = -1;
    most_pixels = 0
    for bin_i in range(0,lumma_bins_n+1):
        n_pixels = np.count_nonzero(lumma_binned == bin_i)
        #print("bin = " + str(bin_i) + "   n_pixels = " + str(n_pixels))
        if n_pixels > most_pixels:
            most_pixels = n_pixels
            most_pixels_bin = bin_i
    
    print("\nlumma_binned: most_pixels_bin = " + str(most_pixels_bin) + "   most_pixels = " + str(most_pixels))
    
    # find background
    
    background_bin = most_pixels_bin
    bg1 = lumma_binned == background_bin
    imshow(bg1, "bg1")
    bg2 = morphology.remove_small_objects(bg1, 5000)
    imshow(bg2, "bg2")
    background = morphology.remove_small_holes(bg2, 10000)
    print('background found')
    imshow(background, "background")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Lumma Background Mask.png")
        cv2.imwrite(filename, background.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    ## Apply the background mask to image
    #bg3 = background.astype(np.uint8) * 255
    #background_img = cv2.bitwise_and(img_rgb, img_rgb, mask = bg3)
    #imshow(background_img, "background_img")
    #filename = os.path.join(output_dir, image_name + " Lumma Background Image.png")
    #cv2.imwrite(filename, background_img, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    
    #background_grown = pixels_within_distance(background, 20)
    
    
    
    
    # find stroma
    
    # 3 bin prior to background bin
    stroma_bin = background_bin-1
    stroma1 = lumma_binned == stroma_bin
    # add two more bin behind
    stroma1 = stroma1 + (lumma_binned == stroma_bin-1)
    stroma1 = stroma1 + (lumma_binned == stroma_bin-2)      # newly added
    # add bins 8 to 0
    stroma1 = stroma1 + (lumma_binned <= 8)                 # newly added
    imshow(stroma1, "stroma1")
    # remove background pixels from stroma
    stroma2 = stroma1 * np.invert(background)
    imshow(stroma2, "stroma2")
    # remove pixels nearby background pixels from stroma
    #stroma3 = stroma2 * np.invert(background_grown)
    #imshow(stroma3, "stroma3")
    # dilation
    stroma4 = morphology.dilation(stroma2, morphology.square(4))    # previously used: 20
    imshow(stroma4, "stroma4")
    # remove small objects
    stroma5 = morphology.remove_small_objects(stroma4, 10000)
    imshow(stroma5, "stroma5")
    # remove small holes
    stroma6 = morphology.remove_small_holes(stroma5, 10000)         # previously used: 20000
    # dilation
    stroma7 = morphology.dilation(stroma6, morphology.square(6))    # previously used: 10
    # remove small holes
    stroma = morphology.remove_small_holes(stroma7, 20000)
    print('stroma found')
    imshow(stroma, "stroma")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Lumma Stroma Mask.png")
        cv2.imwrite(filename, stroma.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    # Apply the stroma mask to image
    stroma6 = stroma.astype(np.uint8) * 255
    stroma_img = cv2.bitwise_and(img_rgb,img_rgb,mask = stroma6)
    imshow(stroma_img, "stroma_img")
    filename = os.path.join(output_dir, image_name + " Stroma.png")
    cv2.imwrite(filename, cv2.cvtColor(stroma_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
    print(filename + " saved")
    
    
    # Find Epithelia
    
    # Anything not Background and not Stroma is Epithelia
    epithelia1 = np.full(lumma_binned.shape, True)
    ## bin prior to Stroma bin
    #epithelia_bin = stroma_bin-2
    ## choosing three bins for epithelia, one bin under stroma
    #epithelia1 = lumma_binned <= epithelia_bin
    #imshow(epithelia1, "epithelia1")
    #epithelia11 = lumma_binned > background_bin
    #epithelia12 = epithelia11 + epithelia1
    #imshow(epithelia12, "epithelia1")
    # remove background pixels from stroma
    epithelia2 = epithelia1 * np.invert(background)
    # remove stroma pixels from stroma
    epithelia = epithelia2 * np.invert(stroma)
    print('epithelia found')
    imshow(epithelia, "epithelia")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Lumma Epithelia Mask.png")
        cv2.imwrite(filename, epithelia.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    # Apply the epithelia mask to image
    epithelia_m = epithelia.astype(np.uint8) * 255
    epithelia_img = cv2.bitwise_and(img_rgb,img_rgb,mask = epithelia_m)
    imshow(epithelia_img, "epithelia_img")
    filename = os.path.join(output_dir, image_name + " Epithelia.png")
    cv2.imwrite(filename, cv2.cvtColor(epithelia_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
    print(filename + " saved")
    
    
    # Save a segmentation visualization image
    if save_rgb_stroma_epithelia_comparison:
        fig, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, 12))
        fig.suptitle('Input - Stroma - Epithelia Segmentation Visualization', fontsize = 25)
        ax1, ax2, ax3 = ax_arr.ravel()
        ax1.imshow(img_rgb)
        ax1.set_title('img_rgb', fontsize = 20)
        ax1.set_axis_off()
        ax2.imshow(stroma_img)
        ax2.set_title('stroma_img', fontsize = 20)
        ax2.set_axis_off()
        ax3.imshow(epithelia_img)
        ax3.set_title('epithelia_img', fontsize = 20)
        ax3.set_axis_off()
        plt.tight_layout()
        filename = os.path.join(output_visualization_dir, image_name + ".png")
        plt.savefig(filename)
        print(filename + " saved")
        if show_images:
            plt.show()
        
        plt.close()
        del fig
    
    if run_over_all_images == 0:
        break;

print("done!")
exit()

