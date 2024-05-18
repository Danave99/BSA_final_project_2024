from functions import *

####################################################################################################################
#################################             Loading the Data              ########################################
####################################################################################################################

Zstack = "Zstack_for_project.tif"

# Load the TIFF stack
with tifffile.TiffFile(Zstack) as tif:
    stack = tif.asarray()

# Determine the indices for the first, last, and middle images
first_image_index = 0
last_image_index = stack.shape[0] - 1
middle_image_index = stack.shape[0] // 2


# Select the first, last, and middle images
first_image = stack[first_image_index]
last_image = stack[last_image_index]
middle_image = stack[middle_image_index]
images_to_process = [first_image, middle_image, last_image]

####################################################################################################################
#################################             Noise Reduction              #########################################
####################################################################################################################

# Processing the Zstack through multiple phases of noise reduction.
noise_reduced_stack = noise_reduction_pipeline(stack)

# Moving average of the zstack.
summed_stack = stack_sum(noise_reduced_stack)

# Removing small spots and removing holes again after applying the moving average
processed_stack = stack_remove_small_objects(summed_stack)

# Eqalized stack is used here to create the brighter images seen later.
# This output is not used for the segmentation, as it yielded worse results.
equalized_stack = stack_hist_eq(processed_stack)

####################################################################################################################
#################################               Segmentation               #########################################
####################################################################################################################



# We start with using Otsu thresholding.
zstack_binary_mask = zstack_mask_otsu(processed_stack)

# We remove even larger "blobs" from the stack. To close gaps inside blood vessels, and remove large areas of noise.
small_blobs_removed = remove_small_blobs(zstack_binary_mask, 80)

# After removing noise, we can multiply the mask by the old stack to get the remaining area.
small_blobs_removed_zstack = processed_stack * small_blobs_removed

# Here we have 2 different kinds of segmentation

# Otsu-Based-Connectivity-Segmentation
segmented_Labels = stack_seg(small_blobs_removed_zstack)

# Watershed Segmentation
watershed_Labels = watershed_segmentation_3d(small_blobs_removed_zstack)

####################################################################################################################
#################################             Skeletonization              #########################################
####################################################################################################################
# Skeletonization based on the watershed Segmentation.
skeletonize_watershed, Skeletonization_watershed_labels = Sekeltonization_pipline_watershed(summed_stack)

# Skeletonization based on the Otsu-Connectivity Segmentation.
skeletonize_otsu_seg_top3, Skeletonization_otsu_labels_top3 = Sekeltonization_pipline_stack_seg(summed_stack,3, False)
skeletonize_otsu_seg_first2, Skeletonization_otsu_labels_first2 = Sekeltonization_pipline_stack_seg(summed_stack, 3, True)



####################################################################################################################
#################################             Graph Creation              #########################################
####################################################################################################################

plt.figure(figsize=(7, 7))
# Plot histogram for grayscale image
plt.hist(middle_image.ravel(), bins=256, color='b', alpha=0.5, label='Grayscale Image')
plt.title('Histogram: Grayscale Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(7, 7))
# Plot histogram for grayscale image
plt.hist(middle_image.ravel(), bins=256, color='b', alpha=0.5, label='Grayscale Image')
plt.title('Histogram: Grayscale Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Compute the magnitude of the Fourier Transform for visualization
magnitude_spectrum_initial = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft2(middle_image))))
# Plot the magnitude spectrum
plt.figure(figsize=(7, 7))
plt.imshow(magnitude_spectrum_initial, cmap='gray')
plt.title('Magnitude Spectrum initial')
plt.colorbar(label='dB')

# Processed image - pre-processing
Processed_image = smooth_image(first_image)


f_image = central_line_filter(Processed_image)
f_image = diagonal_line_filter(np.fft.ifft2(np.fft.ifftshift(f_image)))

# Compute the inverse Fourier Transform to get back to the spatial domain
Processed_image = np.fft.ifft2(np.fft.ifftshift(f_image))
Processed_image = np.abs(Processed_image)
cur_best_Processed_image = low_pass_filter(Processed_image,255)
cur_best_Processed_image = cv2.convertScaleAbs(cur_best_Processed_image, alpha=(255.0/cur_best_Processed_image.max()))
new_processed_image = cv2.medianBlur(cur_best_Processed_image, 3)
# Plot the processed image

plt.figure(figsize=(7, 7))
# Plot histogram for grayscale image
plt.hist(new_processed_image.ravel(), bins=256, color='b', alpha=0.5, label='Grayscale Image')
plt.title('Histogram: Grayscale Image')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

ret_trig, triangle_threshold = cv2.threshold(new_processed_image, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

# Apply Otsu's thresholding
ret_otsu, otsu_threshold = cv2.threshold(new_processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Apply Isodata thresholding
ret_iso, isodata_threshold = cv2.threshold(new_processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Display the original image and the thresholded images side by side
fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(cv2.cvtColor(new_processed_image, cv2.COLOR_GRAY2RGB), cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(triangle_threshold, cmap='gray')
axs[1].set_title('Triangle Threshold')
axs[2].imshow(otsu_threshold, cmap='gray')
axs[2].set_title('Otsu Threshold')
axs[3].imshow(isodata_threshold, cmap='gray')
axs[3].set_title('Isodata Threshold')
plt.tight_layout()
plt.show()

thresh_images = [triangle_threshold,otsu_threshold,isodata_threshold]

for image in thresh_images:
    # Apply remove_small_holes
    min_size = 1  # Minimum size of holes to be removed
    processed_image = remove_small_holes(image, area_threshold=500)
    removed_objects = remove_small_objects(processed_image, min_size=10)
    # Display the original and processed images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(processed_image, cmap='gray')
    axs[1].set_title('After Removing Small Holes')
    axs[2].imshow(removed_objects, cmap='gray')
    axs[2].set_title('After Removing Small Objects')
    plt.tight_layout()
    plt.show()

    # Display the original and adjusted images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(removed_objects, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image, cmap='gray')
    plt.title('Adjusted Image')

    plt.show()

    plt.figure(figsize=(15, 15))
    plt.imshow(new_processed_image * removed_objects, cmap='gray')

    plt.title('Original Image')

    plt.figure(figsize=(7, 7))
    plt.imshow(cur_best_Processed_image, cmap='gray')
    plt.axis('off')
    plt.title(f"Previous Best Processed Image")

    plt.figure(figsize=(7, 7))
    plt.imshow(np.abs(new_processed_image), cmap='gray')
    plt.axis('off')
    plt.title(f"New Processed Image")

    magnitude_spectrum = 20 * np.log10(np.abs(f_image))
    # Plot the magnitude spectrum
    plt.figure(figsize=(7, 7))
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.colorbar(label='dB')

    plt.show()


####################################################################################################################
##############################             Napari Interactive Graphs              ##################################
####################################################################################################################

# Start the Napari Viewer
viewer = napari.Viewer()

# Show the Bright, Processed and Pre-Processed stacks
viewer.add_image(stack, name='Stack_Pre_Processing', colormap='green')
viewer.add_image(processed_stack, name='Stack_Post_Processing', colormap='green')
viewer.add_image(equalized_stack, name='2D_Images_Stack_Post_Processing', colormap='green')

# Show the segmentation. The labels are taken from the Otsu Connectivity segmentation.
# The watershed segmentation failed to create viable labels, however it did create a better image.
viewer.add_image(watershed_Labels[0], name='Watershed_Segmentation', colormap='gray')
viewer.add_labels(segmented_Labels, name='Segmentation_Labels')

# Show the Skeletonization based on the Otsu-Connectivity segmentation 3+ labels
viewer.add_image(skeletonize_otsu_seg_top3, name='Otsu_skeletonization_3plus', colormap='gray')
viewer.add_labels(Skeletonization_otsu_labels_top3, name='Otsu_skeletonization_3plus_labels')
# Show the Skeletonization based on the Otsu-Connectivity segmentation for the main first 2 labels.
viewer.add_image(skeletonize_otsu_seg_first2, name='Otsu_skeletonization_first2', colormap='gray')
viewer.add_labels(Skeletonization_otsu_labels_first2, name='Otsu_skeletonization_first2_Labels')

# Show the Skeletonization based on the Watershed segmentation.
viewer.add_image(skeletonize_watershed, name='Skeletonization with watershed', colormap='gray')
viewer.add_labels(Skeletonization_watershed_labels, name='Skeletonization_Watershed_Labels')

# Run the Napari gui.
napari.run()