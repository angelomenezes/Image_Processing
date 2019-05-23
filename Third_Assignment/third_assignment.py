# Student: Angelo Garangau Menezes
# USP ID: 11413492
# Course Code: SCC0251/SCC5830 - Image Processing
# Assignment 3 : Image Restoration

import numpy as np
import imageio
from scipy.fftpack import fftn, ifftn, fftshift
#import matplotlib.pyplot as plt
#%matplotlib inline


class third_assignment:

    def __init__(self, filename_deg, type_of_filter, parameter_gamma, parameter_size):
        self.filename_deg = filename_deg
        self.type_of_filter = type_of_filter
        self.parameter_gamma = parameter_gamma
        self.parameter_size = parameter_size

    def __call__(self):

        if not self.check_filter_size():
            raise Exception("Please choose a valid size for the filter.")

        if self.type_of_filter == 1:
            return self.denoising()
        elif self.type_of_filter == 2:
            return self.deblurring()

    def denoising(self):

        # Load degraded image
        image = imageio.imread(self.filename_deg)

        padding = int((self.parameter_size - 1)/2)
        reshaped_image = np.pad(image, ((padding, padding),(padding, padding)), mode='constant') # Gotta check if constant is filled with zeros
        filtered_image = []
        center_mask = self.create_mask_center_value(self.parameter_size) # Mask to get central value of matrix
        temp_matrix = np.zeros([self.parameter_size, self.parameter_size])
        center_pixel, fitered_value, centr_l, disp_l = 0,0,0,0

        mode = input()

        if mode == 'average': # Mean and Standard Deviation for centrality and dispersion measurements
            disp_n = self.check_dispersion_n(np.std(reshaped_image[0:(reshaped_image.shape[0]//6 - 1), 0:(reshaped_image.shape[1]//6 - 1)])) # Check general variance of pixels in the image through std

            for row in range(reshaped_image.shape[0] - self.parameter_size + 1):
                for column in range(reshaped_image.shape[1] - self.parameter_size + 1):
                    temp_matrix = reshaped_image[row:self.parameter_size+row, column:self.parameter_size+column]
                    centr_l = temp_matrix.mean()
                    disp_l = self.check_dispersion_l(temp_matrix.std(), disp_n)
                    center_pixel = np.sum(temp_matrix*center_mask) # Gets the pixel of the degraded image (center of the matrix)
                    filtered_value = center_pixel - self.parameter_gamma * (disp_n / disp_l) * (center_pixel - centr_l)
                    filtered_image.append(filtered_value)

        elif mode == 'robust': # Median and Interquatile Range for centrality and dispersion measurements
            disp_n = self.get_interquatile_range(reshaped_image[0:(reshaped_image.shape[0]//6 - 1), 0:(reshaped_image.shape[1]//6 - 1)]) # Check general variance of pixels in the image through Interquatile Range

            for row in range(reshaped_image.shape[0] - self.parameter_size + 1):
                for column in range(reshaped_image.shape[1] - self.parameter_size + 1):
                    temp_matrix = reshaped_image[row:self.parameter_size+row, column:self.parameter_size+column]
                    centr_l = np.median(temp_matrix)
                    disp_l = self.check_dispersion_l(self.get_interquatile_range(temp_matrix), disp_n) # Interquatile Range
                    center_pixel = np.sum(temp_matrix*center_mask) # Gets the pixel of the degraded image (center of the matrix)
                    filtered_value = center_pixel - self.parameter_gamma * (disp_n / disp_l) * (center_pixel - centr_l)
                    filtered_image.append(filtered_value)

        filtered_image = np.array(filtered_image).reshape(image.shape[0], image.shape[1])

        filtered_image = self.normalization(filtered_image, image)

        return filtered_image

    def deblurring(self):

        # Load sigma for gaussian filter
        sigma = float(input())
        sigma = self.check_sigma(sigma)

        # Load degraded image
        image = imageio.imread(self.filename_deg)

        # Laplacian Operator
        laplacian_op = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])

        # Padding Operator
        pad = int(image.shape[0]//2 - laplacian_op.shape[0]//2)
        px_pad = np.pad(laplacian_op, (pad,pad-1), 'constant', constant_values=(0))

        # Degradation Function
        h_deg = self.gaussian_filter(self.parameter_size,sigma)

        # Padding Degradation Matrix
        pad = int(image.shape[0]//2 - h_deg.shape[0]//2)
        H_pad = np.pad(h_deg, (pad,pad-1), 'constant', constant_values=(0))

        # Computing the Fourier transforms
        G_deg = fftn(image)
        H_U = fftn(H_pad)
        P_U = fftn(px_pad)

        # Calculating the CLS function
        filtered_image = np.multiply((H_U.conjugate() / np.abs(H_U)**2 + self.parameter_gamma * np.abs(P_U) ** 2), G_deg)

        # Passing it for the spatial domain
        filtered_image = fftshift(ifftn(filtered_image).real)

        # Normalization based on input data
        filtered_image = self.normalization(filtered_image, image)

        return filtered_image

    def gaussian_filter(self, k=3, sigma=1.0): # We assume the degradation function for the deblurring is a gaussian filter
        arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
        x, y = np.meshgrid(arx, arx)
        filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
        return filt / np.sum(filt)

    def check_filter_size(self):
        if self.parameter_size in [3,5,7,9,11]:
            return True
        return False

    def check_sigma(self, sigma):
        if sigma > 0:
            return sigma
        raise Exception("Please choose a valid sigma.")

    def check_dispersion_n(self, value):
        if value == 0:
            return 1
        return value

    def check_dispersion_l(self, value, disp_n):
        if value == 0:
            return disp_n
        return value

    def create_mask_center_value(self, size):
        center = size//2
        mask = np.zeros([size, size])
        mask[center, center] = 1
        return mask

    def get_interquatile_range(self, matrix):
        percentiles = np.percentile(matrix, [75, 25])
        return percentiles[0] - percentiles[1]

    def normalization(self,image, reference):
        min_ = np.min(image)
        max_ = np.max(image)
        quantized_image = np.max(reference)*(((image - min_)/(max_ - min_)))
        return quantized_image

# Function that calculates how far the images are from the testing set
def RMSE(image1, image2):
    image1 = image1.astype(float)
    image2 = image2.astype(float)
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

    # User input: (Filename of the reference image, Filename of degraded image, Type of filter, Parameter Gamma, Size of filter)
    filename_ref = str(input()).rstrip()
    filename_deg = str(input()).rstrip()

    type_of_filter = int(input())
    gamma = float(input())
    size_of_filter = int(input())

    restored_image = third_assignment(filename_deg, type_of_filter, gamma, size_of_filter)()
    reference_image = imageio.imread(filename_ref)

    #comparing_images(final_image, test_image)
    print("%.3f" % RMSE(restored_image, reference_image))
