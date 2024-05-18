# Importing libraries is important.
# Please Make sure you have downloaded all the necessary libraries to run this code.
# To make sure you have done so, "pip install _____" everything that is needed before claiming something does not work.

import numpy as np
np.bool = np.bool_
import matplotlib.pyplot as plt
import cv2
import tifffile
from PIL import Image
import napari
from napari.utils import nbscreenshot

from scipy import ndimage
from scipy import signal
from scipy import linalg
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

from skimage import io
from skimage import color
from skimage import morphology
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters import threshold_otsu, threshold_isodata, threshold_triangle
from skimage import filters, measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.io import imread
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu, threshold_triangle, threshold_isodata
from skimage import morphology
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters import threshold_otsu, threshold_isodata, threshold_triangle


####################################################################################################################
##############################             Noise Reduction Functions             ###################################
####################################################################################################################

# Runs a Gaussian filter over the image, smoothing out the edges.
def smooth_image(image, kernel_size=(3, 3)):
    return cv2.GaussianBlur(image, kernel_size, 1)


# Cuts off the high frequencies, keeping the low ones.
# We found that cutting any more than the 2 highest frequencies caused the image to degrade very rapidly.
def low_pass_filter(image, cutoff_frequency):
    # Move the image to the frequency domain.
    f = fft2(image)
    fshift = fftshift(f)

    # Create a mask used to cut off the outer frequencies.
    mask = np.ones_like(fshift)
    mask[cutoff_frequency:-cutoff_frequency, cutoff_frequency:-cutoff_frequency] = 0

    # Multiply the image by the mask to remove the unwanted frequencies.
    fshift_masked = fshift * mask

    # Shift the result back into a regular image.
    f_ishift = ifftshift(fshift_masked)
    img_back = np.abs(ifft2(f_ishift))
    return img_back


# Unwanted frequencies filter.
# After we found the main noise frequencies in the image (4 bright spots near the center of the frequency domain,
# 4 more near the edges of the Y axis, and a line through the center), this function cuts those frequencies out.
# Notably, these noise frequencies remain generally the same throughout the stack.
def central_line_filter(image):

    # Create an array of the FFT of the image.
    f_image = np.fft.fft2(image)
    fshift = np.fft.fftshift(f_image)
    image_gray_fft2 = fshift.copy()

    # Reduce the brightness of specific areas of the fft of the image so that the noise frequencies are not longer visible.
    image_gray_fft2[:256, fshift.shape[1] // 2] = 1
    image_gray_fft2[-255:, fshift.shape[1] // 2] = 1
    image_gray_fft2[-246:-236, -246:-236] = 1
    image_gray_fft2[236:246, 236:246] = 1
    image_gray_fft2[236:246, -246:-236] = 1
    image_gray_fft2[-246:-236, 236:246] = 1
    image_gray_fft2[-20:-10, 236:246] = 1
    image_gray_fft2[-20:-10, -246:-236] = 1
    image_gray_fft2[10:20, 236:246] = 1
    image_gray_fft2[10:20, -246:-236] = 1

    return image_gray_fft2


# Is there a reason this does nothing but convert the image into an FFT?
def diagonal_line_filter(image):
    f_image = np.fft.fft2(image)
    fshift = np.fft.fftshift(f_image)
    image_gray_fft2 = fshift.copy()

    return image_gray_fft2

# Function that creates an otsu mask of the zstack, which can be reused for many other purposes.
def zstack_mask_otsu(zstack):
    # We copy the zstack onto another variable, to not change the original variable during the function.
    data_3d = zstack

    # Reshape the data to 2D for Otsu's thresholding
    data_2d = data_3d.reshape(-1, 1).astype(np.uint8)

    # Perform Otsu's thresholding
    _, thresholded_img = cv2.threshold(data_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Reshape the thresholded image back to 3D
    binary_mask = thresholded_img.reshape(data_3d.shape)
    return binary_mask

# Function for removing small objects from the stack in its entirety, instead of each image separately.
def remove_small_blobs(binary_mask: np.ndarray, min_size: int = 0):

    #Removes from the input mask all the blobs having less than N adjacent pixels.
    #We set the small objects to the background label 0.

    if min_size > 0:
        dtype = binary_mask.dtype
        binary_mask = morphology.remove_small_objects(binary_mask.astype(bool), min_size=min_size)
        binary_mask = binary_mask.astype(dtype)
    return binary_mask


# Function for creating a moving average of images across the entire stack.
def stack_sum(zstack):
    summed_stack = np.zeros(zstack.shape)

    # We add adjacent images together, then average their sum, to get the average of adjacent images (Or a moving average).
    # This helps the stack stay in a coherent shape as we move from the first image to the last, and helps the segmentation look better.
    for i in range(zstack.shape[0]):
        if i == 0:
            summed_stack[i,:,:] = (zstack[i,:,:] + zstack[i+1,:,:]) / 2
        elif i > 0 and i < zstack.shape[0] - 1:
            summed_stack[i, :, :] = (zstack[i-1,:,:] + zstack[i,:,:] + zstack[i + 1,:,:]) / 3

        elif i == zstack.shape[0] - 1:
            summed_stack[i, :, :] = (zstack[i,:,:] + zstack[i-1,:,:]) / 2

    return summed_stack

# running a histogram equalization on each image in a stack.
# This part helped the images look better when seen by a human eye, however it was not used for the segmentation itself.
# It was used for the skeletonization, as for this part, we wanted to create a mask that would be more easily connected with the skeletonization algorithm.
def stack_hist_eq(zstack):
    equalized_stack = np.zeros(zstack.shape)

    for i in range(zstack.shape[0]):
        cur_img = zstack[i,:,:]

        # Make sure the image is in uint8
        cur_img = cv2.convertScaleAbs(cur_img, alpha=(255.0 / cur_img.max()))

        # Run the histogram equalization
        equalized_image = cv2.equalizeHist(cur_img)
        equalized_stack[i,:,:] = equalized_image

    return equalized_stack

#Function used to remove small objects, and small holes from each image in a stack. Name is slightly misleading
def stack_remove_small_objects(zstack, area_thresh=50, min_size=30):
    processed_stack = np.zeros(zstack.shape)

    for i in range(zstack.shape[0]):
        cur_img = zstack[i,:,:]
        thresh_image = cur_img > 0
        # Run remove small holes -> remove small objects
        removed_holes = remove_small_holes(thresh_image, area_threshold=area_thresh)
        removed_objects = remove_small_objects(removed_holes, min_size=min_size)

        # Multiply the image by the mask
        final_image = removed_objects * cur_img
        processed_stack[i,:,:] = final_image


    return processed_stack


# Function to dilate the objects in each image in a stack, with variable possible kernel size. This version was used, while the erosion wasn't.
def zstack_dialation(zstack , kernel_size = (3,3)):

    dilated_stack = np.zeros(zstack.shape)

    for i in range(zstack.shape[0]):
        cur_img = zstack[i,:,:]

        # Create the structuring element.
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        # Perform Dilation
        dilated_image = cv2.dilate(cur_img, structuring_element)
        dilated_stack[i,:,:] = dilated_image
    return dilated_stack


########################         Main Noise Reduction Function (The Pipeline)        ################################
# The noise reduction pipeline of the stack, incorporates many other functions, detailed within.
def noise_reduction_pipeline(zstack, rank=100):
    noise_reduced_zstack = np.zeros(shape=zstack.shape)
    # Function runs on each image in the stack of 85 images.
    for i in range(zstack.shape[0]):
        cur_img = zstack[i, :, :]
        # Run the Gaussian filter to smooth the image.
        Processed_image = smooth_image(cur_img)

        # Remove noise frequencies.
        f_image = central_line_filter(Processed_image)
        f_image = diagonal_line_filter(np.fft.ifft2(np.fft.ifftshift(f_image)))

        # Compute the inverse Fourier Transform to get back to the spatial domain
        Processed_image = np.fft.ifft2(np.fft.ifftshift(f_image))
        Processed_image = np.abs(Processed_image)

        # Use SVD to smooth the images further. We found the best rank for this was 100.
        U, s, V = linalg.svd(Processed_image)

        cur_sigmas = s.copy()
        cur_sigmas[rank:] = 0  # Subset of the ranks
        S = linalg.diagsvd(cur_sigmas, U.shape[1], V.shape[1])
        Processed_image = U @ S @ V  # Recreate the images

        # Move the image through a low pass filter to smooth it even further.
        # Here we only removed the highest frequency 256. As we found that removing even 255 completely ruined the image.
        Processed_image = low_pass_filter(Processed_image, 255)

        # The images were strangely not in uint8 in the stack, it's here we normalize them.
        Processed_image = cv2.convertScaleAbs(Processed_image, alpha=(255.0 / Processed_image.max()))

        # Apply a Median Blur to help get rid of salt and pepper noise.
        Processed_image = cv2.medianBlur(Processed_image, 3)

        # Use Otsu Thresholding to get rid of more noise. (We tested separately that this one gave us the best results).
        ret_iso, isodata_threshold = cv2.threshold(Processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # In order to close gaps inside of arteries, and to remove spots, we dilated the image, then eroded it twice, then dilated it again.
        # This closes any 1 pixel holes, and also removes and 1 pixel noise left in the image.
        isodata_threshold = morphology.binary_closing(isodata_threshold)
        isodata_threshold = morphology.binary_opening(isodata_threshold)

        # Here we remove holes in the image. This part is not depth dependent, as when the data gets deeper, we generally get more white noise than black noise.
        # Meaning that as we go deeper, there are more small objects, rather than more holes.
        removed_holes = remove_small_holes(isodata_threshold, area_threshold=50)

        # This part is depth dependent, so that as we go deeper into the stack, we remove larger and larger objects.
        # This is because the deeper into the stack, the more white noise there is, and the larger the noise becomes.
        # Also, at the later depths, the main bodies of arteries are generally much larger as can be seen in the images.
        if 0 <= i < 30:
            removed_objects = remove_small_objects(removed_holes, min_size=30)
        elif 30 <= i < 60:
            removed_objects = remove_small_objects(removed_holes, min_size=50)
        elif 60 <= 60 <= 85:
            removed_objects = remove_small_objects(removed_holes, min_size=70)

        # Finally we multiply the image by the final mask we created.
        final_image = Processed_image * removed_objects

        # We add the image to the previously empty stack.
        noise_reduced_zstack[i, :, :] = final_image

    return noise_reduced_zstack



####################################################################################################################
#################################               Segmentation               #########################################
####################################################################################################################

# This function Segments a Z-stack of images using skimage.
def stack_seg(zstack):
    # We copy the zstack onto another variable, to not change the original variable during the function.
    data_3d = zstack

    # Reshape the data to 2D for Otsu's thresholding
    data_2d = data_3d.reshape(-1, 1).astype(np.uint8)

    # Perform Otsu's thresholding
    _, thresholded_img = cv2.threshold(data_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Reshape the thresholded image back to 3D
    binary_mask = thresholded_img.reshape(data_3d.shape)


    # Perform 3D connected component analysis
    labels = measure.label(binary_mask, connectivity=3)

    return labels


# Function to perform watershed segmentation on a single image
def watershed_segmentation(image):

    image = cv2.convertScaleAbs(image, alpha=(255.0 / image.max()))

    # Convert the image to 8-bit 3-channel format
    image = cv2.merge((image, image, image))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to obtain a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed algorithm
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries

    return [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), markers]


# Function to segment a z-stack of images using the previous Watershed Algorithm
def watershed_segmentation_3d(zstack):
    segmented_stack = np.zeros_like(zstack)
    labels_stack = np.zeros_like(zstack)
    for i in range(zstack.shape[0]):
        cur_image = zstack[i, :, :]
        segmented_image = watershed_segmentation(cur_image)
        labels_stack[i, :, :] = segmented_image[1]
        segmented_stack[i, :, :] = segmented_image[0]

    return [segmented_stack, np.uint8(cv2.convertScaleAbs(labels_stack, alpha=(255.0 / labels_stack.max())))]



####################################################################################################################
#################################             Skeletonization              #########################################
####################################################################################################################

# For easier readability, a function for using the skimage.morphology.skeletonize function on a zstack, using the 3d lee94 method.
def skeleton_stack(segmented_zstack):
    return morphology.skeletonize(segmented_zstack, method='lee')


#Skeletonization_pipeline using the watershed_3d function
def Sekeltonization_pipline_watershed(zstack):
    equalized_stack = stack_hist_eq(zstack)

    masked_stack = zstack_mask_otsu(equalized_stack)

    skel_processed_stack = remove_small_holes(masked_stack, area_threshold=80)
    small_blobs_removed_skel = remove_small_blobs(skel_processed_stack, 80)

    pre_skeletonization_segmentation = zstack_dialation(np.uint8(small_blobs_removed_skel), kernel_size=(9, 9))

    small_blobs_removed_skel_stack = np.uint8(pre_skeletonization_segmentation)
    skel_segmented_zstack = watershed_segmentation_3d(small_blobs_removed_skel_stack)
    segmented_binary_mask = zstack_mask_otsu(skel_segmented_zstack[0])

    skeletonize_zstack = skeleton_stack(segmented_binary_mask)
    Skeletonization_labels = stack_seg(skeletonize_zstack)


    return [skeletonize_zstack, Skeletonization_labels]

#Skeletonization_pipeline using the stack_seg function
def Sekeltonization_pipline_stack_seg(zstack, min_label = 3, show_net = False):
    equalized_stack = stack_hist_eq(zstack)

    masked_stack = zstack_mask_otsu(equalized_stack)

    skel_processed_stack = remove_small_holes(masked_stack, area_threshold=80)
    small_blobs_removed_skel = remove_small_blobs(skel_processed_stack, 80)

    pre_skeletonization_segmentation = zstack_dialation(np.uint8(small_blobs_removed_skel), kernel_size=(9, 9))

    small_blobs_removed_skel_stack = np.uint8(pre_skeletonization_segmentation)
    skel_segmented_zstack = stack_seg(small_blobs_removed_skel_stack)
    if show_net == False:
        segmented_binary_mask = skel_segmented_zstack >= min_label

    elif show_net == True:
        segmented_binary_mask =  np.select([skel_segmented_zstack==0, skel_segmented_zstack == 1, skel_segmented_zstack ==2], [np.zeros_like(skel_segmented_zstack), np.ones_like(skel_segmented_zstack), 2*np.ones_like(skel_segmented_zstack)])

    else:
        skel_segmented_zstack = small_blobs_removed_skel_stack

    skeletonize_zstack = skeleton_stack(segmented_binary_mask)
    Skeletonization_labels = stack_seg(skeletonize_zstack)

    return [skeletonize_zstack, Skeletonization_labels]



########################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################
##########################################################                         Unused Functions                          ##########################################################################
########################################################################################################################################################################################################################################
########################################################################################################################################################################################################################################
########################################### Only wild functions that either partially worked, or were deemed unnecessary for the rest of the code.                ######################################################################
########################################### They are only documented as far as was needed when they were used, with a short background or description of each.    ######################################################################
########################################################################################################################################################################################################################################
###########################################                          Tread carefully beyond this point, and view at your own risk ;)                              ######################################################################
########################################################################################################################################################################################################################################



#################################          Noise Reduction And Morphology           ###################################


# There ended up being easier ways to load the zstack, this ended up being a useless function.
def load_tiff(Zstack):
    try:
        # Load the TIFF stack using skimage.io.imread
        im = io.imread(Zstack)
        print("Stack dimensions are:", im.shape)

        # Use PIL.Image to handle TIFF stack and extract frames
        dataset = Image.open(Zstack)
        h, w = dataset.size
        tiffarray = np.zeros((h, w, dataset.n_frames))

        for i in range(dataset.n_frames):
            dataset.seek(i)
            tiffarray[:, :, i] = np.array(dataset)

        # Convert to double precision
        expim = tiffarray.astype(np.double)
        print("Processed stack dimensions are:", expim.shape)

        return expim  # Return the processed TIFF stack as a NumPy array
    except Exception as e:
        print("Error loading TIFF stack:", e)
        return None  # Return None if there's an error


# Because we ended up mostly working on the entire zstack all at once, slicing was not necessary.
def slicing(Zstack):
    zstack = io.imread(Zstack)
    # Determine the total number of slices in the Z-stack
    total_slices = zstack.shape[0]
    print(total_slices)

    # Calculate the indices for 17 evenly spaced slices
    indices = np.linspace(0, total_slices - 1, 17, dtype=int)

    # Extract the selected slices
    selected_slices = zstack[indices, :, :]
    return(selected_slices)


# We ended up finding more convenient ways of presenting the images.
def present_slices(image_list):
    # Calculate the number of rows and columns for the grid
    # Assuming you want to display 17 images in a grid
    total_images = len(image_list)
    columns = 5 # Adjusted to fit 17 images
    rows = total_images // columns + (total_images % columns > 0)

    # Create a figure
    fig = plt.figure(figsize=(17, 10))

    # Loop through the depth slices and add them to the figure
    for i in range(total_images):
        # Calculate the row and column indices for the subplot
        row = i // columns
        col = i % columns

        # Add a subplot at the calculated position
        ax = fig.add_subplot(rows, columns, i+1)

        # Display the depth slice
        ax.imshow(image_list[i], cmap='gray') # Adjust 'cmap' as needed
        ax.axis('off') # Hide axes
        ax.set_title(f"Depth Slice {i+1}") # Use the loop index directly

    # Adjust the spacing between subplots to create gaps
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Show the figure
    plt.show()


# We ended up using a better version of this kind of function.
def remove_salt_pepper_noise(image, kernel_size=3):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8))
    return image

# We ended up using a better version of this kind of function.
def remove_small_spots(image, kernel_size=(3,3)):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# Ended up being unnecessary and overly-complex. The noise frequencies were mostly the same at all depths, and so there was no reason to specifically adjust them.
def adjust_cutoff_frequency(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Calculate the magnitude of the Fourier Transform
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # Create a mask to exclude the center of the spectrum (DC component)
    rows, cols = dft_shift.shape[:2]
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # Apply the mask to exclude the DC component
    masked_spectrum = cv2.bitwise_and(magnitude_spectrum, magnitude_spectrum, mask=mask)

    # Find the peak frequency
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(masked_spectrum)

    # Calculate the cutoff frequency based on the peak frequency
    # This is a simple heuristic; you might need to adjust this based on your specific images
    cutoff_frequency = maxLoc[0] if maxLoc[0] > 30 else 30

    return cutoff_frequency

# Ended up being unnecessary and overly-complex. The noise frequencies were mostly the same at all depths, and so there was no reason to specifically detect them.
def detect_diagonal_frequencies(image):
    # Perform Fourier Transform on the image
    f_transform = fftshift(fft2(image))

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(f_transform)

    # Find peaks in the magnitude spectrum excluding the center
    peaks = cv2.minMaxLoc(magnitude_spectrum)[3]
    center_x, center_y = image.shape[0] // 2, image.shape[1] // 2
    peaks = [peak for peak in peaks if peak != (center_x, center_y)]

    return peaks


# Function to erode the objects in each image in a stack, with variable possible kernel size.
def stack_erosion(zstack , kernel_size = (3,3)):

    eroded_stack = np.zeros(zstack.shape)

    for i in range(zstack.shape[0]):
        cur_img = zstack[i,:,:]
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

        # Perform erosion
        eroded_image = cv2.erode(cur_img, structuring_element)
        eroded_stack[i,:,:] = eroded_image


    return eroded_stack


# Function for using the skimage.morphology.binary_closing on each image in a zstack
def stack_closing(zstack):
    closed_stack = np.zeros(zstack.shape)
    for i in range(zstack.shape[0]):
        cur_img = zstack[i,:,:]
        closed_image = morphology.binary_closing(cur_img)
        closed_stack[i,:,:] = closed_image * cur_img
    return closed_stack

# Function for using the skimage.morphology.binary_opening on each image in a zstack
def stack_opening(zstack):
    open_stack = np.zeros(zstack.shape)
    for i in range(zstack.shape[0]):
        cur_img = zstack[i,:,:]
        opened_image = morphology.binary_opening(cur_img)
        open_stack[i,:,:] = opened_image * cur_img

    return open_stack


# Function for using SVD on each image in a stack.
def stack_SVD(zstack, rank=100):
    labels_stack = np.zeros(zstack.shape)

    for i in range(zstack.shape[0]):
        cur_img = zstack[i, :, :]

        U, s, V = linalg.svd(cur_img)

        cur_sigmas = s.copy()
        cur_sigmas[rank:] = 0  # Subset of the ranks
        S = linalg.diagsvd(cur_sigmas, U.shape[1], V.shape[1])
        approx_blood_vessels = U @ S @ V  # Recreate the owl



        labels_stack[i, :, :] = approx_blood_vessels

    return labels_stack




#################################             Segmentation              ########################################





# Functions that were supposed to help view the segmentations:
# We found better methods of observing the results. Mainly Napari

#     View the segmented Z-stack in a 3D plot.
def view_segmented_images(segmented_stack):

    # Generate colors for each label
    unique_labels = np.unique(segmented_stack)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    # Broadcast the colors array to match the shape of the segmented_stack
    colors = np.repeat(colors[np.newaxis, :], segmented_stack.shape[0], axis=0)

    # Visualize the segmented objects in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(segmented_stack, facecolors=colors)
    plt.show()


# Display segmented images from a 3D segmentation stack.
def show_segmented_images(segmented_stack):
    num_slices = segmented_stack.shape[0]

    # Display each segmented slice
    for i in range(num_slices):
        plt.figure()
        plt.imshow(segmented_stack[i, :, :], cmap='nipy_spectral')
        plt.title(f"Segmented Image - Slice {i}")
        plt.axis('off')
        plt.show()




#################################             Skeletonization              ########################################

# Function to skeletonize a single segmented image - We ended up not using this function, as it skeletonized in 2D
# which ended up not being effective when applied to the zstack
def skeletonize_image(image):
    # Convert the image to binary
    ret, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

    # Apply skeletonization
    skel = np.zeros_like(binary_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(binary_image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary_image, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary_image = eroded.copy()

        zeros = len(np.argwhere(binary_image == 255))
        if zeros == 0:
            done = True

    return skel