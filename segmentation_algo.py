
##########################################

##       SEGMENTATION ALGORITHM         ##

kShowImages = 0                                 # to display images in real-time
kSaveIntermediateImages = 0                    # save R-G-B and Y-Cb-Cr Channels
#kInputDir = "data/sheffield_h&e"               # all files in here will be read, expected filenames: <filename>.tif
#kInputDir = "data/liverpool_h&e"
kInputDir = "data_test"
#kOutputDir = "extracted/sheffield_h&e"         # epithelia and stroma will be saved here
#kOutputDir = "extracted/liverpool_h&e"
#kOutputDir = "workingdir/segmented"
kOutputDir = "output_test"
kSaveEpitheliaAndStroma = 0                   # to save epithelia and stroma in output dir (not needed during development)
kRunOverAllImages = 0                         # to run over all images in 'kInputDir'
kOverwriteOutput = 0                            # to overwrite previous output
#image_name = "test"             # specific image to run with 'kRunOverAllImages = 0'
image_name = "h2114186 h&e_ROI_3"
#image_name = "h2114186 h&e_ROI_3"
#image_name = "h1810898B  h&e_ROI_4"
#image_name = "h2114155 h&e_ROI_4"

# visualizations

#kOutputVisualizationDir = "extracted/sheffield_h&e/visualization3"         # output for visual comparison b/w input_img-segmented_stroma-segmented_epithelia
#kOutputVisualizationDir = "extracted/liverpool_h&e/visualization3"
kOutputVisualizationDir = kOutputDir
kRescaleSize = 0.3                          # Downscaling visualization image to make our development life easy
kSaveBinsRepresentation = 0                    # to save Lumma and Red Chroma Bins for visualization

# computing resources
kPercentMinMemoryAvailableToStartNewThread = 70     # if execution gets killed because of low memory, increase this constant. If you have > 32G then you can decrease this

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
from skimage import color
from skimage import morphology
from skimage import measure
#from skimage import segmentation
#from skimage import filters
from skimage.transform import rescale
#import scipy.ndimage as ndimage
import gc
import psutil
import time
import math
from bounded_pool_executor import BoundedProcessPoolExecutor   # https://github.com/mowshon/bounded_pool_executor/tree/master

program_start_time = time.time()

# figure id - reusing matplotlib figures to save memory
kFigureId = 1

def ProgramRunTime():
    seconds = int(time.time() - program_start_time)
    minutes = math.floor(seconds / 60)
    seconds = seconds % 60
    runtime = f"({minutes:02}:{seconds:02})"
    return runtime


def CurrentIndex():
    global current_file_index, total_num_of_images
    return f"({current_file_index}/{total_num_of_images})"

def GetPercentAvailableMemory():
    # Get the virtual memory statistics
    mem = psutil.virtual_memory()
    # Get the available memory
    #total_memory_gb = mem.total / (1024 * 1024 * 1024)
    #available_memory_gb = mem.available / (1024 * 1024 * 1024)
    #print(f"Memory: {available_memory_gb:.1f}GB/{total_memory_gb:.1f}GB")
    percent_mem_available = int(mem.available / mem.total * 100)
    print(f"{ProgramRunTime()} {CurrentIndex()} Memory Available = {percent_mem_available}%")
    return percent_mem_available

# Plot the image
def imshow(img, title):
    if kShowImages:
        plt.imshow(img)
        plt.tight_layout()
        plt.axis('off')
        plt.title(title)
        plt.show(block=True)
        plt.close('all')

#def pixels_within_distance(mask, distance):
#    """Finds pixels within a specified distance from a 2D mask."""
#    # Create a distance map from the mask
#    distance_map = ndimage.distance_transform_edt(np.logical_not(mask))
#    # Find pixels within the specified distance
#    pixels_within = distance_map <= distance
#    return pixels_within

# arguments must be in the specified order, matching regionprops
def image_stdev(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.std(intensities[region], ddof=1)

# arguments must be in the specified order, matching regionprops
def image_var(region, intensities):
    # note the ddof arg to get the sample var if you so desire!
    return np.var(intensities[region], ddof=1)

def show_images_side_by_side(images, image_labels, combined_label):
    global kSaveBinsRepresentation, image_name, kOutputVisualizationDir, kShowImages
    no_of_images = len(images)
    fig = plt.figure(num=kFigureId, clear=True, figsize=(20, 12))
    ax_arr = fig.subplots(1, no_of_images, sharex=True, sharey=True)
    fig.suptitle(image_name + " " + combined_label, fontsize = 25)
    col=0
    for img_no in range(0,no_of_images):
        ax_arr[col].set_title(image_labels[img_no], fontsize = 20)
        ax_arr[col].set_axis_off()
        ax_arr[col].imshow(images[img_no])
        col=col+1
    plt.tight_layout()
    if kSaveBinsRepresentation:
        filename = os.path.join(kOutputVisualizationDir, image_name, image_name + " " + combined_label + ".png")
        plt.savefig(filename)
        print(filename + " saved")
    if kShowImages:
        plt.show()

def create_binned_representation(img_2d, label, no_of_bins = 20, bins_on_plot = 20, first_bin_on_plot = 0):
    # decimate lumma image into lumma_bins_n
    img_binned = np.clip((np.floor(img_2d.astype(np.float64) * no_of_bins / 255)).astype(np.uint8), 0, no_of_bins-1)
    ## figure to show different img_2d bins
    if kSaveBinsRepresentation:
        fig = plt.figure(num=kFigureId, clear=True, figsize=(20, 12))
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
        filename = os.path.join(kOutputVisualizationDir, image_name, image_name + " " + label + " " + str(no_of_bins) + " Bins Representation.png")
        plt.savefig(filename)
        print(filename + " saved")
        if kShowImages:
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
    global files, total_num_of_images
    # Note: don't add 'image_name' to above list of global variables.
    
    if file_index != -1:
        f = files[file_index]
        #if kRunOverAllImages:
        print("\n*********** CURRENT FILE (" + str(file_index+1) + "/" + str(total_num_of_images) + "): " + f)
        x = f.split(".", 1)
        if x[1] != "tif":
            print(x)
            print("Unexpected Filename:" + f + " - more dots than anticipated. Exiting...")
            exit()
        image_name = x[0]
        del x
    
    if kOverwriteOutput == 0 and kRunOverAllImages:
        if os.path.exists(os.path.join(kOutputVisualizationDir, image_name + ".png")):
            return
    
    input_filepath = os.path.join(kInputDir, image_name)
    print(input_filepath + ".tif")
    
    # Load Image
    img_rgb = cv2.imread(input_filepath + ".tif")
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    
    # time mapping
    #start_time = time.perf_counter_ns()
    #end_time = time.perf_counter_ns()
    #print(f"RGB Channels Runtime: {((end_time - start_time) / 1000)} microseconds")
    
    # representation bins and all visualization files will be saved here
    if kSaveBinsRepresentation:
        if not os.path.isdir(os.path.join(kOutputVisualizationDir, image_name)):
            os.mkdir(os.path.join(kOutputVisualizationDir, image_name))
            print(os.path.join(kOutputVisualizationDir, image_name) + " directory created.")
    
    if kSaveBinsRepresentation or kShowImages:
        show_images_side_by_side([img_rgb, img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]], ["RGB", "R", "G", "B"], "RGB Channels")
    
    img_YCrCb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    # Split the image into its channels
    img_Y, img_Cr, img_Cb = cv2.split(img_YCrCb)
    #imshow(img_YCrCb, 'img_YCrCb')
    del img_YCrCb
    
    if kSaveBinsRepresentation or kShowImages:
        show_images_side_by_side([img_rgb, img_Y, img_Cb, img_Cr], ["RGB", "Y", "Cb", "Cr"], "YCbCr Channels")
    
    #img_Cb_binned, img_Cb_most_pixels_bin = save_binned_representation(img_Cb, "Blue Chroma")
    #img_Cr_binned, img_Cr_most_pixels_bin = save_binned_representation(img_Cr, "Red Chroma")
    
    # YUV
    
    img_yuv_f = color.rgb2yuv(img_rgb, channel_axis=-1)
    if kSaveBinsRepresentation or kShowImages:
        show_images_side_by_side([img_rgb, img_yuv_f[:,:,0], img_yuv_f[:,:,1], img_yuv_f[:,:,2]], ["RGB", "Y", "U", "V"], "YUV Channels")
    #save_binned_representation((np.clip(((img_yuv_f[:,:,1] + 1) / 2 * 255), 0, 255)).astype(np.uint8), "U")
    #save_binned_representation((np.clip(((img_yuv_f[:,:,2] + 1) / 2 * 255), 0, 255)).astype(np.uint8), "V")
    
    #kShowImages = 1
    #kSaveBinsRepresentation = 1
    img_u = np.clip(((img_yuv_f[:,:,1] + 1) / 2 * 255), 0, 255)
    imshow(img_u, "img_u_f")
    img_v = np.clip(((img_yuv_f[:,:,2] + 1) / 2 * 255), 0, 255)
    imshow(img_v, "img_v_f")
    img_u_expanded = (np.clip(((img_yuv_f[:,:,1] * 4 + 1) / 2 * 255), 0, 255)).astype(np.uint8)
    imshow(img_u_expanded, "img_u_expanded")
    img_v_expanded = (np.clip(((img_yuv_f[:,:,2] * 4 + 1) / 2 * 255), 0, 255)).astype(np.uint8)
    imshow(img_v_expanded, "img_v_expanded")
    
    if kSaveBinsRepresentation or kShowImages:
        create_binned_representation(img_u, "img_u")
        create_binned_representation(img_v, "img_v")
        create_binned_representation(img_u_expanded, "img_u_expanded")
        create_binned_representation(img_v_expanded, "img_v_expanded")
    
    # Locate Background
    
    # we will get background from lumma channel
    
    lumma_binned, lumma_most_pixels_bin = create_binned_representation(img_Y, "Lumma")
    
    del img_yuv_f, img_u, img_Y, img_Cr
    
    background_bin = lumma_most_pixels_bin
    background = lumma_binned == background_bin
    imshow(background, "bg1 bin=" + str(lumma_most_pixels_bin))
    background = morphology.remove_small_objects(background, 5000)
    imshow(background, "bg2")
    background = morphology.remove_small_holes(background, 10000)
    print('background located')
    imshow(background, "background")
    if kSaveIntermediateImages:
        filename = os.path.join(kOutputDir, image_name + " Lumma Background Mask.png")
        cv2.imwrite(filename, background.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    ## Apply the background mask to image
    #background_img = cv2.bitwise_and(img_rgb, img_rgb, mask = (background.astype(np.uint8) * 255))
    #imshow(background_img, "background_img")
    #filename = os.path.join(kOutputDir, image_name + " Lumma Background Image.png")
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
    imshow(epithelia, "epithelia")
    # label connected areas in epithelia
    labels = measure.label(epithelia)
    img_v_dialated = morphology.dilation(img_v, morphology.square(5))
    region_props = measure.regionprops(labels, img_v_dialated, extra_properties=[image_stdev, image_var])
    # get average red chroma value in areas
    for prop in region_props:
        print(f"label={prop.label} mean={prop.intensity_mean} stdev={prop.image_stdev} var={prop.image_var}")
        if prop.intensity_mean < 133:
            # invert areas having mean intensity lower than light red
            print(f"remove area label={prop.label} with img_v_dialated mean={prop.intensity_mean}")
            epithelia = epithelia * np.invert(labels == prop.label)
    print('epithelia located')
    if kSaveIntermediateImages:
        filename = os.path.join(kOutputDir, image_name + " Epithelia Mask.png")
        cv2.imwrite(filename, epithelia.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    del labels, region_props, img_v
    
    
    # Apply the epithelia mask to image
    epithelia_img = cv2.bitwise_and(img_rgb,img_rgb,mask = (epithelia.astype(np.uint8) * 255))
    imshow(epithelia_img, "epithelia_img")
    if kSaveEpitheliaAndStroma:
        filename = os.path.join(kOutputDir, image_name + " Epithelia.png")
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
    if kSaveIntermediateImages:
        filename = os.path.join(kOutputDir, image_name + " Stroma Mask.png")
        cv2.imwrite(filename, stroma.astype(np.uint8)*255, [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    # Apply the stroma mask to image
    stroma_img = cv2.bitwise_and(img_rgb,img_rgb,mask = (stroma.astype(np.uint8) * 255))
    imshow(stroma_img, "stroma_img")
    if kSaveEpitheliaAndStroma:
        filename = os.path.join(kOutputDir, image_name + " Stroma.png")
        cv2.imwrite(filename, cv2.cvtColor(stroma_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
        print(filename + " saved")
    
    # Save a segmentation visualization image
    # Combine the images horizontally
    stroma_img_rescaled = (rescale(stroma_img, kRescaleSize, anti_aliasing=True, channel_axis=2) * 255).astype(np.uint8)
    epithelia_img_rescaled = (rescale(epithelia_img, kRescaleSize, anti_aliasing=True, channel_axis=2) * 255).astype(np.uint8)
    img_rgb_rescaled = (rescale(img_rgb, kRescaleSize, anti_aliasing=True, channel_axis=2) * 255).astype(np.uint8)
    white_border = np.ones((img_rgb_rescaled.shape[0], 100, 3), dtype=np.uint8) * 255    # White Border - 3 layer
    combined_image = np.hstack((img_rgb_rescaled, white_border, stroma_img_rescaled, white_border, epithelia_img_rescaled), dtype=np.uint8)
    imshow(combined_image, 'combined_image')
    filename = os.path.join(kOutputVisualizationDir, image_name + ".png")
    cv2.imwrite(filename, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION , 0])
    print(filename + " saved")
    
    del stroma_img_rescaled, epithelia_img_rescaled, img_rgb_rescaled, combined_image, white_border
    del img_rgb, img_Cb, stroma_img, epithelia_img, background, epithelia, stroma, stroma2, lumma_binned, filename, img_u_expanded_binned_50, img_v_expanded_binned_50
    # Force a garbage collection
    gc.collect()





if not os.path.isdir(kInputDir):
    print("kInputDir: '" + kInputDir + "' directory does not exist! Exiting...")
    exit()

if not os.path.isdir(kOutputDir):
    os.mkdir(kOutputDir)
    print("kOutputDir: '" + kOutputDir + "' directory created.")

if not os.path.isdir(kOutputVisualizationDir):
    os.mkdir(kOutputVisualizationDir)
    print("kOutputVisualizationDir: '" + kOutputVisualizationDir + "' directory created.")
    
# Files and Folders in Input Dir
files = os.listdir(kInputDir)
# Filtering only the files.
files = [f for f in files if os.path.isfile(kInputDir+'/'+f)]
total_num_of_images = len(files)
print(f"Image Dataset size: {total_num_of_images}")
current_file_index = 0

# Get available cpu cores
print("Number of CPU cores:", os.cpu_count())
no_of_threads_to_use = os.cpu_count() - 2
print("Number of Threads to use:", no_of_threads_to_use)

# Get the virtual memory statistics
percent_mem_available = GetPercentAvailableMemory()

if kRunOverAllImages == 1 and kSaveBinsRepresentation == 0 and kShowImages == 0:
    # use multi threading
    with BoundedProcessPoolExecutor(max_workers=no_of_threads_to_use) as worker:
        for file_index in range(total_num_of_images):
            current_file_index = file_index
            percent_mem_available = GetPercentAvailableMemory()
            while percent_mem_available < kPercentMinMemoryAvailableToStartNewThread:
                time.sleep(1)
                percent_mem_available = GetPercentAvailableMemory()
            print('#%d Worker initialization' % file_index)
            worker.submit(segmentation_algo, file_index)
            time.sleep(0.1)
else:
    # run one by one
    if kRunOverAllImages == 1:
        for file_index in range(total_num_of_images):
            f = files[file_index]
            x = f.split(".", 1)
            if x[1] != "tif":
                print(x)
                print("Unexpected Filename:" + f + " - more dots than anticipated. Exiting...")
                exit()
            image_name = x[0]
            del x
            segmentation_algo(file_index)
    else:
        segmentation_algo(-1, image_name)


print(f"{ProgramRunTime()} {CurrentIndex()} done!")
exit()

