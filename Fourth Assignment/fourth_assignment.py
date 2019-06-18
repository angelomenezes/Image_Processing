# Student: Angelo Garangau Menezes
# USP ID: 11413492
# Course Code: SCC0251/SCC5830 - Image Processing
# Assignment 4 : Color Image Segmentation

import numpy as np
import random
import imageio
#import matplotlib.pyplot as plt
#%matplotlib inline

class fourth_assignment:

    def __init__(self, filename, type_of_attribute, n_clusters, n_iterations, seed):
        self.filename = filename
        self.type_of_attribute = type_of_attribute
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.seed = seed

    def __call__(self):
        return self.kmeans()

    def attribute_parser(self, image):

        if self.type_of_attribute == 1: # (R,G,B)
            RGB = np.array(image).reshape(image.shape[0]*image.shape[1], 3)

        elif self.type_of_attribute == 2: # (R, G, B, x, y)
            RGB = np.array(image).reshape(image.shape[0]*image.shape[1], 3)
            x = np.array([i for i in range(image.shape[0]) for j in range(image.shape[1])]).reshape(image.shape[0]*image.shape[1],1)
            y = np.array([j for i in range(image.shape[0]) for j in range(image.shape[1])]).reshape(image.shape[0]*image.shape[1],1)
            axes = np.append(y, x, axis=1)
            RGB = np.append(RGB, axes, axis=1)

        elif self.type_of_attribute == 3: # (Luminance)
            luminance = []
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    value = 0.299 * image[i,j,0] + 0.587 * image[i,j,1] + 0.114 * image[i,j,2]
                    luminance.append(value)
            RGB = np.array(luminance).reshape(image.shape[0]*image.shape[1],1)

        elif self.type_of_attribute == 4: # (Luminance, x, y)
            luminance = []
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    value = 0.299 * image[i,j,0] + 0.587 * image[i,j,1] + 0.114 * image[i,j,2]
                    luminance.append(value)
            RGB = np.array(luminance).reshape(image.shape[0]*image.shape[1],1)

            x = np.array([i for i in range(image.shape[0]) for j in range(image.shape[1])]).reshape(image.shape[0]*image.shape[1],1)
            y = np.array([j for i in range(image.shape[0]) for j in range(image.shape[1])]).reshape(image.shape[0]*image.shape[1],1)
            axes = np.append(y, x, axis=1)
            RGB = np.append(RGB, axes, axis=1)

        return RGB

    def kmeans(self):

        image_orig = imageio.imread(self.filename)

        img_shape = image_orig.shape[0]*image_orig.shape[1]
        image = self.attribute_parser(image_orig)

        random.seed(self.seed)
        ids = np.sort(random.sample(range(0, img_shape), self.n_clusters))
        cluster_centroids = image[ids] # Centroid initialization

        #while(len(np.unique(cluster_centroids, axis=0)) != len(cluster_centroids)): # This can be added to make sure there was not any repeated cluster centroid
        #    ids = np.sort(random.sample(range(0, img_shape), self.n_clusters))
        #    cluster_centroids = image[ids]

        for _ in range(self.n_iterations):

            distance_vec = np.sqrt(((image - cluster_centroids[:, np.newaxis])**2).sum(axis=2))
            closest_clusters = np.argmin(distance_vec, axis=0) # Array containing the index to the nearest centroid for each point
            cluster_centroids = np.nan_to_num(np.array([image[closest_clusters==k].mean(axis=0) for k in range(self.n_clusters)])) # Updating cluster centroids
            #print(np.unique(closest_clusters, return_counts=True))

        return self.normalization(closest_clusters.reshape(image_orig.shape[0],image_orig.shape[1]))


    def normalization(self, image):
        min_ = np.min(image)
        max_ = np.max(image)
        quantized_image = 255*(((image - min_)/(max_ - min_)))
        return quantized_image

# Function that calculates how far the images are from the testing set

def RMSE(image1, image2):
    image1 = image1.astype(float)
    image2 = image2.astype(float)
    return np.float(np.sqrt(((image1 - image2)**2).mean()))

if __name__  == "__main__":

    # User input: (Filename of input image, Filename of reference image, Type of attribute, Number of Clusters, Number of Iterations, Seed)
    filename_input = str(input()).rstrip()
    filename_ref = str(input()).rstrip()

    type_of_attribute = int(input())
    n_clusters = int(input())
    n_iterations = int(input())
    seed = int(input())

    segmented_image = fourth_assignment(filename_input, type_of_attribute, n_clusters, n_iterations, seed)()
    reference_image = np.load(filename_ref)

    print("%.4f" % RMSE(segmented_image, reference_image))
