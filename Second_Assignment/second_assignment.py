# Student: Angelo Garangau Menezes
# USP ID: 11413492
# Course Code: SCC0251/SCC5830 - Image Processing
# Assignment 2 : Image Enhancement and Filtering

# Implementation of 4 functions:
# 1 - Limiarization (Adaptive)
# 2 - 1D Filtering
# 3 - 2D Filtering with Limiarization
# 4 - 2D Median Filter

import numpy as np
import random
import imageio
import matplotlib.pyplot as plt

# Method 1
def limiarization(image, initial_threshold):

    filtered_image = np.zeros([image.shape[0], image.shape[1]])
    actual_threshold = initial_threshold
    diff = 1

    while(diff > 0.5):

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j] > actual_threshold:
                    filtered_image[i,j] = 1
                else:
                    filtered_image[i,j] = 0

        group_of_1 = filtered_image*image # Getting group of pixels that passed the threshold
        group_of_0 = (1-filtered_image)*image # Checking group of pixels that did not pass the threshold

        new_threshold = (np.sum(group_of_1)/np.sum(filtered_image) + np.sum(group_of_0)/np.sum(1-filtered_image))/2 # Calculating new threshold

        diff = np.abs(new_threshold - actual_threshold)
        actual_threshold = new_threshold

    filtered_image = normalization(filtered_image) # Normalizing the image

    return filtered_image

# Method 2
def filter_1D(image, size_filter, weights):

    reshaped_image = image.reshape(-1)
    padding = int((size_filter - 1)/2) # Calculating the number of rows/columns to be padded, now limited to functions where total number of dim needs to be even/odd depending on original image
    reshaped_image = np.pad(reshaped_image, (padding,padding), mode='symmetric') # Padding with Symmetry

    filtered_image = []

    for index in range(len(reshaped_image) - size_filter + 1):
        filtered_image.append(np.sum(reshaped_image[index:size_filter+index] * weights)) # Conv-1D

    filtered_image = np.array(filtered_image).reshape(image.shape[0], image.shape[1]) # Reshaping to original dimension

    filtered_image = normalization(filtered_image) # Normalizing the image

    return filtered_image

# Method 3
def filter_2D(image, size_filter, weights, threshold):

    padding = int((size_filter - 1)/2) # Calculating the number of rows/columns to be padded, now limited to functions where total number of dim needs to be even/odd depending on original image
    reshaped_image = np.pad(image, ((padding, padding),(padding, padding)), mode='symmetric') # Padding with Symmetry
    filtered_image = []

    for row in range(reshaped_image.shape[0] - size_filter + 1):
        for column in range(reshaped_image.shape[1] - size_filter + 1):
            filtered_image.append(np.sum(reshaped_image[row:size_filter+row, column:size_filter+column] * weights)) # Point-wise multiplication between matrices (Conv-2D)

    filtered_image = np.array(filtered_image).reshape(image.shape[0], image.shape[1]) # Organizing the matrix in the same shape of the original image

    filtered_image = limiarization(filtered_image, threshold) # Limiarization + Normalization

    return filtered_image

# Method 4
def median_filter_2D(image, size_filter):

    padding = int((size_filter - 1)/2) # Calculating the number of rows/columns to be padded, now limited to functions where total number of dim needs to be even/odd depending on original image
    reshaped_image = np.pad(image, ((padding, padding),(padding, padding)), mode='constant') # Padding with Zeros
    filtered_image = []

    for row in range(reshaped_image.shape[0] - size_filter + 1):
        for column in range(reshaped_image.shape[1] - size_filter + 1):
            filtered_image.append(np.median(reshaped_image[row:size_filter+row, column:size_filter+column])) # Calculating median

    filtered_image = np.array(filtered_image).reshape(image.shape[0], image.shape[1]) # Organizing the matrix in the same shape of the original image

    filtered_image = normalization(filtered_image)

    return filtered_image

# Normalizing to 0-255
def normalization(image):
    min_ = np.min(image)
    max_ = np.max(image)
    quantized_image = (((image - min_)/(max_ - min_))*255)
    return quantized_image

# Function that calculates how far the images are from the testing set
def RMSE(image1, image2):
    return np.float(np.sqrt(((image1 - image2)**2).mean()))

# A simple function for visualization
def comparing_images(image1, image2):
    _ = plt.figure(figsize=(5,5))
    _ = plt.subplot(1,2,1)
    _ = plt.imshow(image1, cmap='gray')
    _ = plt.subplot(1,2,2)
    _ = plt.imshow(image2, cmap='gray')
    plt.show()

if __name__  == "__main__":

    # User input: (Filename of the reference image, method and specific parameters to each method)

    filename = str(input()).rstrip()
    method = int(input())

    test_image = imageio.imread(filename)

    if method == 1:

        threshold = int(input())
        final_image = limiarization(test_image, threshold)

    elif method == 2:

        weights = []
        size_filter = int(input())
        str_weights = str(input())
        str_weights = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in str_weights) # Converting a string line into an array of weights
        weights = np.array([float(i) for i in str_weights.split()])
        final_image = filter_1D(test_image, size_filter, weights)

    elif method == 3:

        weights = []
        size_filter = int(input())
        for _ in range(size_filter): # Converting several strings into a single matrix of weights
            str_weights = str(input())
            str_weights = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in str_weights)
            weights.append([float(i) for i in str_weights.split()])
        weights = np.array(weights).reshape(size_filter, size_filter)
        threshold = int(input())
        final_image = filter_2D(test_image, size_filter, weights, threshold)

    elif method == 4:
        size_filter = int(input())
        final_image = median_filter_2D(test_image, size_filter)

    else:
        print('Method was not recognized.')

    #comparing_images(final_image, test_image)
    print("%.4f" % RMSE(final_image, test_image))
