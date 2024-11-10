
##########################################

##       SEGMENTATION ALGORITHM         ##

show_images = 0                                 # to display images in real-time
save_intermediate_images = 0                    # save R-G-B and Y-Cb-Cr Channels
input_dir = "data/sheffield_h&e"               # all files in here will be read, expected filenames: <filename>.tif
#input_dir = "data/liverpool_h&e"
#input_dir = "data_test"
output_dir = "extracted/sheffield_h&e"         # epithelia and stroma will be saved here
#output_dir = "extracted/liverpool_h&e"
#output_dir = "workingdir/segmented"
#output_dir = "output_test"
run_over_all_images = 1                         # to run over all images in 'input_dir'
overwrite_output = 0                            # to overwrite previous output
#image_name = "test"             # specific image to run with 'run_over_all_images = 0'
#image_name = "h2114158 h&e_ROI_2"
#image_name = "h2114186 h&e_ROI_3"
image_name = "h2114182 h&e_ROI_3"
#image_name = "h2114155 h&e_ROI_4"
save_epithelia_and_stroma = 0                   # to save epithelia and stroma output

# visualizations

output_visualization_dir = "extracted/sheffield_h&e/visualization"         # output for visual comparison b/w input_img-segmented_stroma-segmented_epithelia
#output_visualization_dir = "extracted/liverpool_h&e/visualization"
#output_visualization_dir = output_dir
#rescale_size = 0.5
save_bins_representation = 0                    # to save Lumma and Red Chroma Bins for visualization

# computing resources
max_no_of_threads_to_use = 6
# Recommended RAM > 27Gb
# If app gets killed by low memory, reduce max_no_of_threads_to_use, or increase it if you have more RAM and CPU Cores
# If app gets killed, make 'overwrite_output = 0' to start from where it left last time

##########################################


## Import OpenCV

import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
import cv2
import numpy as np
#import matplotlib 
#matplotlib.use('Agg') # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import os
#from skimage import data
from skimage import color
from skimage import morphology
#from skimage import segmentation
#from skimage import filters
from skimage.transform import rescale
import scipy.ndimage as ndimage
import gc
import threading
import psutil
import concurrent.futures
import time

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
    plt.close('all')

def pixels_within_distance(mask, distance):
    """Finds pixels within a specified distance from a 2D mask."""
    # Create a distance map from the mask
    distance_map = ndimage.distance_transform_edt(np.logical_not(mask))
    # Find pixels within the specified distance
    pixels_within = distance_map <= distance
    return pixels_within

def show_images_side_by_side(images, image_labels, combined_label):
    global save_bins_representation, figure_id, image_name, output_visualization_dir, show_images
    no_of_images = len(images)
    fig = plt.figure(num=figure_id, clear=True, figsize=(20, 12))
    ax_arr = fig.subplots(1, no_of_images, sharex=True, sharey=True)
    fig.suptitle(image_name + " " + combined_label, fontsize = 25)
    col=0
    for img_no in range(0,no_of_images):
        ax_arr[col].set_title(image_labels[img_no], fontsize = 20)
        ax_arr[col].set_axis_off()
        ax_arr[col].imshow(images[img_no])
        col=col+1
    plt.tight_layout()
    if save_bins_representation:
        filename = os.path.join(output_visualization_dir, image_name + " " + combined_label + ".png")
        plt.savefig(filename)
        print(filename + " saved")
    if show_images:
        plt.show()

def create_binned_representation(img_2d, label, no_of_bins = 20, bins_on_plot = 20, first_bin_on_plot = 0):
    global save_bins_representation, figure_id, image_name, output_visualization_dir, show_images
    # decimate lumma image into lumma_bins_n
    img_binned = np.clip((np.floor(img_2d.astype(np.float64) * no_of_bins / 255)).astype(np.uint8), 0, no_of_bins-1)
    ## figure to show different img_2d bins
    if save_bins_representation:
        fig = plt.figure(num=figure_id, clear=True, figsize=(20, 12))
        ax_arr = fig.subplots(2, int(bins_on_plot/2), sharex=True, sharey=True)
        fig.suptitle(image_name + " " + label + " " + str(no_of_bins) + " Bins Representation", fontsize = 25)
        row=0
        col=0
        for bin_i in range(0,bins_on_plot):
            ax_arr[row,col].set_title("bin " + str(bin_i + first_bin_on_plot), fontsize = 20)
            ax_arr[row,col].set_axis_off()
            ax_arr[row,col].imshow(img_binned == (bin_i + first_bin_on_plot))
            col=col+1
            if col == int(bins_on_plot/2):
                row = 1
                col = 0
        plt.tight_layout()
        filename = os.path.join(output_visualization_dir, image_name + " " + label + " " + str(no_of_bins) + " Bins Representation.png")
        plt.savefig(filename)
        print(filename + " saved")
        if show_images:
            plt.show()
    # find bin with most number of pixels
    most_pixels_bin = -1;
    most_pixels = 0
    for bin_i in range(0,no_of_bins+1):
        n_pixels = np.count_nonzero(img_binned == bin_i)
        #print("bin = " + str(bin_i) + "   n_pixels = " + str(n_pixels))
        if n_pixels > most_pixels:
            most_pixels = n_pixels
            most_pixels_bin = bin_i
    print(label + "_binned: most_pixels_bin = " + str(most_pixels_bin) + "   most_pixels = " + str(most_pixels))
    return img_binned, most_pixels_bin


# the segmentation algo that runs over an image
# input:
#   file_index = index of file in files to work with
def segmentation_algo(file_index, image_name = ""):
    global show_images, save_intermediate_images, input_dir, output_dir, save_epithelia_and_stroma, output_visualization_dir, save_bins_representation, files, overwrite_output, run_over_all_images, total_num_of_images
    
    if file_index != -1:
        f = files[file_index]
        #if run_over_all_images:
        print("\n*********** CURRENT FILE (" + str(file_index+1) + "/" + str(total_num_of_images) + "): " + f)
        x = f.split(".", 1)
        if x[1] != "tif":
            print(x)
            print("Unexpected Filename:" + f + " - more dots than anticipated. Exiting...")
            exit()
        image_name = x[0]
        del x
    
    if overwrite_output == 0 and run_over_all_images:
        if os.path.exists(os.path.join(output_visualization_dir, image_name + ".png")):
            return
    
    input_filepath = os.path.join(input_dir, image_name)
    print(input_filepath + ".tif")
    
    # figure id - reusing figures to save memory
    figure_id = 1 #file_index % no_of_threads_to_use
    
    # Load Image
    img_rgb = cv2.imread(input_filepath + ".tif")
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    imshow(img_rgb, 'img_rgb')
    
    # time mapping
    #start_time = time.perf_counter_ns()
    #end_time = time.perf_counter_ns()
    #print(f"RGB Channels Runtime: {((end_time - start_time) / 1000)} microseconds")
    
    if save_bins_representation or show_images:
        show_images_side_by_side([img_rgb, img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]], ["RGB", "R", "G", "B"], "RGB Channels")
    
    img_YCrCb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    # Split the image into its channels
    img_Y, img_Cr, img_Cb = cv2.split(img_YCrCb)
    #imshow(img_YCrCb, 'img_YCrCb')
    del img_YCrCb
    
    if save_bins_representation or show_images:
        show_images_side_by_side([img_rgb, img_Y, img_Cb, img_Cr], ["RGB", "Y", "Cb", "Cr"], "YCbCr Channels")
    
    #img_Cb_binned, img_Cb_most_pixels_bin = save_binned_representation(img_Cb, "Blue Chroma")
    #img_Cr_binned, img_Cr_most_pixels_bin = save_binned_representation(img_Cr, "Red Chroma")
    
    # YUV
    img_yuv_f = color.rgb2yuv(img_rgb, channel_axis=-1)
    if save_bins_representation or show_images:
        show_images_side_by_side([img_rgb, img_yuv_f[:,:,0], img_yuv_f[:,:,1], img_yuv_f[:,:,2]], ["RGB", "Y", "U", "V"], "YUV Channels")
    #save_binned_representation((np.clip(((img_yuv_f[:,:,1] + 1) / 2 * 255), 0, 255)).astype(np.uint8), "U")
    #save_binned_representation((np.clip(((img_yuv_f[:,:,2] + 1) / 2 * 255), 0, 255)).astype(np.uint8), "V")
    
    #show_images = 1
    #save_bins_representation = 1
    img_u = np.clip(((img_yuv_f[:,:,1] + 1) / 2 * 255), 0, 255)
    imshow(img_u, "img_u_f")
    img_v = np.clip(((img_yuv_f[:,:,2] + 1) / 2 * 255), 0, 255)
    imshow(img_v, "img_v_f")
    img_u_expanded = (np.clip(((img_yuv_f[:,:,1] * 4 + 1) / 2 * 255), 0, 255)).astype(np.uint8)
    imshow(img_u_expanded, "img_u_expanded")
    img_v_expanded = (np.clip(((img_yuv_f[:,:,2] * 4 + 1) / 2 * 255), 0, 255)).astype(np.uint8)
    imshow(img_v_expanded, "img_v_expanded")
    
    if save_bins_representation or show_images:
        create_binned_representation(img_u, "img_u")
        create_binned_representation(img_v, "img_v")
        create_binned_representation(img_u_expanded, "img_u_expanded")
        create_binned_representation(img_v_expanded, "img_v_expanded")
    
    # Locate Background
    
    # we will get background from lumma channel
    
    lumma_binned, lumma_most_pixels_bin = create_binned_representation(img_Y, "Lumma")
    
    del img_yuv_f, img_u, img_v, img_Y, img_Cr
    
    background_bin = lumma_most_pixels_bin
    background = lumma_binned == background_bin
    imshow(background, "bg1 bin=" + str(lumma_most_pixels_bin))
    background = morphology.remove_small_objects(background, 5000)
    imshow(background, "bg2")
    background = morphology.remove_small_holes(background, 10000)
    print('background located')
    imshow(background, "background")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Lumma Background Mask.png")
        cv2.imwrite(filename, background.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    ## Apply the background mask to image
    #background_img = cv2.bitwise_and(img_rgb, img_rgb, mask = (background.astype(np.uint8) * 255))
    #imshow(background_img, "background_img")
    #filename = os.path.join(output_dir, image_name + " Lumma Background Image.png")
    #cv2.imwrite(filename, background_img, [cv2.IMWRITE_PNG_COMPRESSION , 0])
    #del background_img
    
    #background_grown = pixels_within_distance(background, 20)
    
    
    # Bin Blue Chroma channel
    
    # we will get epithelia from Blue Chroma channel
    
    no_of_chroma_bins = 50
    #Cb_binned, Cb_most_pixels_bin = save_binned_representation(img_Cb, "Blue Chroma", no_of_chroma_bins, 10, 20)
    img_u_expanded_binned_50, img_u_expanded_50_most_pixels_bin = create_binned_representation(img_u_expanded, "img_u_expanded", no_of_chroma_bins, 14, 18)
    
    # Bin Red Chroma channel
    #Cr_binned, Cr_most_pixels_bin = save_binned_representation(img_Cr, "Red Chroma", no_of_chroma_bins, 10, 24)
    img_v_expanded_binned_50, img_v_expanded_50_most_pixels_bin = create_binned_representation(img_v_expanded, "img_v_expanded", no_of_chroma_bins, 20, 24)
    
    del img_u_expanded, img_v_expanded
    
    # Locate Definite Stroma
    
    # Definite Stroma is middle bin in img_u_expanded_binned_50
    # and highly red region >= 42 bin in img_v_expanded_binned_50
    stroma_bin = no_of_chroma_bins/2
    stroma = img_u_expanded_binned_50 == stroma_bin
    stroma = stroma + (img_v_expanded_binned_50 >= 42)
    imshow(stroma, "stroma1")
    # remove background pixels from stroma
    stroma2 = stroma * np.invert(background)
    imshow(stroma2, "stroma2")
    # dilation
    stroma = morphology.dilation(stroma2, morphology.square(3))
    imshow(stroma, "stroma3")
    # remove small objects
    stroma = morphology.remove_small_objects(stroma, 1000)
    imshow(stroma, "definite stroma")
    print('definite stroma located')
    
    
    # Locate Epithelia
    
    # 1-line easy approximation: :)
    # imshow((img_Cr >= 137) * (img_Cb <= 145), "epithelia")
    
    # Epithelia is the first 12 bins that are 1 bin ahead of middle in img_v_expanded_binned_50
    # add bins 27-32 in img_u_expanded_binned_50
    # and three Bins from Lumma - background bin / 3 and two below
    epithelia_bin = no_of_chroma_bins / 2 + 1
    epithelia = img_v_expanded_binned_50 >= epithelia_bin
    epithelia = epithelia * np.invert((img_v_expanded_binned_50 >= (epithelia_bin + 12)))
    # add bins 27-32 in img_u_expanded_binned_50
    epithelia = epithelia + (img_u_expanded_binned_50 >= 27)
    epithelia = epithelia * np.invert((img_u_expanded_binned_50 >= 33))
    imshow(epithelia, "epithelia1")
    # remove background pixels from epithelia
    epithelia = epithelia * np.invert(background)
    imshow(epithelia, "epithelia2")
    ## and add three Bins from Lumma - background bin / 3 and two below
    epithelia = epithelia + (lumma_binned == int(background_bin / 3))
    epithelia = epithelia + (lumma_binned == int(background_bin / 3 - 1))
    epithelia = epithelia + (lumma_binned == int(background_bin / 3 - 2))
    imshow(epithelia, "epithelia3lu")
    # remove stroma pixels from epithelia
    epithelia = epithelia * np.invert(stroma)
    imshow(epithelia, "epithelia3")
    # remove blue ink drop region
    # Mark regions with blue ink drop for removal from epithelia
    imshow(img_Cb > 152, "blue_ink")
    epithelia = epithelia * np.invert(img_Cb > 152)
    imshow(epithelia, "epithelia4")
    # dilation
    #epithelia = morphology.dilation(epithelia, morphology.square(3))
    epithelia = morphology.dilation(epithelia, morphology.square(2))
    imshow(epithelia, "epithelia5")
    # remove small holes
    epithelia = morphology.remove_small_holes(epithelia, 10000)
    imshow(epithelia, "epithelia7")
    # remove very small objects
    epithelia = morphology.remove_small_objects(epithelia, 500)
    imshow(epithelia, "epithelia6")
    # find out epithelia information at current state
    n_epithelia_pixels = np.count_nonzero(epithelia)
    percent_epithelia_pixels = n_epithelia_pixels / (epithelia.shape[0] * epithelia.shape[1])
    print("percent_epithelia_pixels = " + str(percent_epithelia_pixels))
    small_obj_size_to_remove = 10000
    if percent_epithelia_pixels < 0.01:
        small_obj_size_to_remove = 1000
    elif percent_epithelia_pixels < 0.03:
        small_obj_size_to_remove = 2000
    print("small_obj_size_to_remove = " + str(small_obj_size_to_remove))
    # remove small objects
    epithelia = morphology.remove_small_objects(epithelia, small_obj_size_to_remove)
    imshow(epithelia, "epithelia8")
    # dilation
    epithelia = morphology.dilation(epithelia, morphology.square(10))
    imshow(epithelia, "epithelia9")
    # remove small holes
    epithelia = morphology.remove_small_holes(epithelia, 20000)
    print('epithelia located')
    imshow(epithelia, "epithelia")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Epithelia Mask.png")
        cv2.imwrite(filename, epithelia.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    # Apply the epithelia mask to image
    epithelia_img = cv2.bitwise_and(img_rgb,img_rgb,mask = (epithelia.astype(np.uint8) * 255))
    imshow(epithelia_img, "epithelia_img")
    if save_epithelia_and_stroma:
        filename = os.path.join(output_dir, image_name + " Epithelia.png")
        cv2.imwrite(filename, cv2.cvtColor(epithelia_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    # Expanded Stroma
    
    # remove small holes
    stroma = morphology.remove_small_holes(stroma2, 10000)
    imshow(stroma, "stroma3")
    # dilation
    stroma = morphology.dilation(stroma, morphology.square(10))
    imshow(stroma, "stroma4")
    # remove small holes
    stroma = morphology.remove_small_holes(stroma, 20000)
    imshow(stroma, "stroma5")
    # remove background pixels from expanded stroma
    stroma = stroma * np.invert(background)
    # remove stroma pixels from expanded stroma
    stroma = stroma * np.invert(epithelia)
    print('stroma expanded')
    imshow(stroma, "stroma")
    if save_intermediate_images:
        filename = os.path.join(output_dir, image_name + " Stroma Mask.png")
        cv2.imwrite(filename, stroma.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    # Apply the stroma mask to image
    stroma_img = cv2.bitwise_and(img_rgb,img_rgb,mask = (stroma.astype(np.uint8) * 255))
    imshow(stroma_img, "stroma_img")
    if save_epithelia_and_stroma:
        filename = os.path.join(output_dir, image_name + " Stroma.png")
        cv2.imwrite(filename, cv2.cvtColor(stroma_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    
    # Save a segmentation visualization image
    # Combine the images horizontally
    #stroma_img_rescaled_f = rescale(stroma_img, rescale_size, anti_aliasing=True, channel_axis=2)
    #epithelia_img_rescaled_f = rescale(epithelia_img, rescale_size, anti_aliasing=True, channel_axis=2)
    #img_rgb_rescaled_f = rescale(img_rgb, rescale_size, anti_aliasing=True, channel_axis=2)
    white_border_f = np.ones((img_rgb.shape[0], 100, 3), dtype=np.uint8) * 255    # White Border - 3 layer
    combined_image = np.hstack((img_rgb, white_border_f, stroma_img, white_border_f, epithelia_img), dtype=np.uint8)
    imshow(combined_image, 'combined_image')
    filename = os.path.join(output_visualization_dir, image_name + ".png")
    cv2.imwrite(filename, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
    print(filename + " saved")
    
    del img_rgb, img_Cb, combined_image, white_border_f, stroma_img, epithelia_img, background, epithelia, stroma, stroma2, lumma_binned, filename, img_u_expanded_binned_50, img_v_expanded_binned_50
    # Force a garbage collection
    gc.collect()





if not os.path.isdir(input_dir):
    print("input_dir: '" + input_dir + "' directory does not exist! Exiting...")
    exit()

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
    print("output_dir: '" + output_dir + "' directory created.")

if not os.path.isdir(output_visualization_dir):
    os.mkdir(output_visualization_dir)
    print("output_visualization_dir: '" + output_visualization_dir + "' directory created.")
    

# Files and Folders in Input Dir
files = os.listdir(input_dir)
# Filtering only the files.
files = [f for f in files if os.path.isfile(input_dir+'/'+f)]
total_num_of_images = len(files)
print(f"Image Dataset size: {total_num_of_images}")

# Get available cpu cores
cores = os.cpu_count()
print("Number of CPU cores:", cores)

no_of_threads_to_use = min(cores - 2, max_no_of_threads_to_use)
print("Number of Threads to use:", no_of_threads_to_use)

# Get the virtual memory statistics
mem = psutil.virtual_memory()
# Get the available memory
available_memory_bytes = mem.available
available_memory_gb = available_memory_bytes / (1024 * 1024 * 1024)
print(f"Available memory: {available_memory_gb:.2f} GB")

if run_over_all_images == 1 and save_bins_representation == 0 and show_images == 0:
    # use multi threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=no_of_threads_to_use) as executor:
        executor.map(segmentation_algo, range(total_num_of_images))
else:
    # run one by one
    if run_over_all_images == 1:
        for f in range(0, total_num_of_images):
            segmentation_algo(f)
    else:
        segmentation_algo(-1, image_name)


print("done!")
exit()

