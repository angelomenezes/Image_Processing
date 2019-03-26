# Student: Angelo Garangau Menezes
# USP ID: Yet to be obtained
# Course Code: SCC0251/SCC5830 - Image Processing
# Assignment 1 : Image Generation

import numpy as np
import random
import matplotlib.pyplot as plt

class first_assignment:
    
    # Parameter Initiliazition 
    def __init__(self, parameter_C, parameter_fun, parameter_Q, parameter_N, parameter_B, parameter_S):
        self.parameter_C = parameter_C
        self.parameter_fun = parameter_fun
        self.parameter_Q = parameter_Q
        self.parameter_N = parameter_N
        self.parameter_B = parameter_B
        self.parameter_S = parameter_S
    
    # Five possible functions that can be specified by the user input  
    def __call__(self):       
        if self.parameter_fun == 1:
            return self.function_1()
        elif self.parameter_fun == 2:
            return self.function_2()
        elif self.parameter_fun == 3:
            return self.function_3()
        elif self.parameter_fun == 4:
            return self.function_4()
        elif self.parameter_fun == 5:
            return self.function_5()
        else:
            raise Exception("Please choose a valid function.")

    # Step for downsampling the image ("digitization")
    def digitize(self, image):
        z = []
        size = (self.parameter_N, self.parameter_N)

        if (len(image) % self.parameter_N == 0):
            for i in range(0, len(image), int(len(image)/self.parameter_N)):
                for j in range(0, len(image), int(len(image)/self.parameter_N)):
                    z.append(image[i][j])
        else:
            for i in range(self.parameter_N):
                for j in range(self.parameter_N):
                    z.append(image[i][j])            
        return np.reshape(np.array(z), size)
    
    # Normalization step to uint16 (0-65535)
    def normalize_float(self, image):
        min_ = np.min(image)
        max_ = np.max(image)
        normalized_image = (((image - min_)/(max_ - min_))*65535).astype(np.uint16)
        return normalized_image
    
    # Quantization step
    def quantization(self, image):
        min_ = np.min(image)
        max_ = np.max(image)
        quantized_image = (((image - min_)/(max_ - min_))*255).astype(np.uint8)
        return quantized_image
    
    # Bit shift step (needed for quantization)
    def bit_shift(self, image):
        return image >> (8-self.parameter_B)
    
    def function_1(self):
        image = np.zeros([self.parameter_C, self.parameter_C])
        for x in range(self.parameter_C):
            for y in range(self.parameter_C):
                image[x,y] = x*y + 2*y
        
        image = self.normalize_float(image)
        return self.bit_shift(self.quantization(self.digitize(image)))
    
    def function_2(self):
        image = np.zeros([self.parameter_C, self.parameter_C])
        for x in range(self.parameter_C):
            for y in range(self.parameter_C):
                image[x,y] = np.abs(np.cos(x/self.parameter_Q) + 2* np.sin(y/self.parameter_Q))
        image = self.normalize_float(image)
        return self.bit_shift(self.quantization(self.digitize(image)))
    
    def function_3(self):
        image = np.zeros([self.parameter_C, self.parameter_C])
        for x in range(self.parameter_C):
            for y in range(self.parameter_C):
                image[x,y] = np.abs(3*(x/self.parameter_Q) - np.cbrt(y/self.parameter_Q))
        image = self.normalize_float(image)
        return self.bit_shift(self.quantization(self.digitize(image)))
    
    def function_4(self):
        random.seed(self.parameter_S)
        image = np.zeros([self.parameter_C, self.parameter_C])
        #image = np.random.randint(0,2,[self.parameter_C,self.parameter_C])
        for x in range(self.parameter_C):
            for y in range(self.parameter_C):
                image[x,y] = random.uniform(0,1)
        image = self.normalize_float(image)
        return self.bit_shift(self.quantization(self.digitize(image)))
    
    def function_5(self):
        random.seed(self.parameter_S)
        image = np.zeros([self.parameter_C, self.parameter_C])
        steps = 1 + self.parameter_C*self.parameter_C
        x,y = 0,0

        for step in range(steps):
            image[x, y] = 1
            dx = random.randint(-1,1)
            dy = random.randint(-1,1)
            x = (x + dx) % self.parameter_C
            y = (y + dy) % self.parameter_C
            image[x, y] = 1

        #image = self.normalize_float(image)
        return self.bit_shift(self.quantization(self.digitize(image)))

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
    
    # User input: (Filename of the reference image, parameter C, parameter Q, parameter N, parameter B, parameter S)
    filename = str(input()).rstrip()
    C = int(input())
    fun = int(input())
    Q = int(input())
    N = int(input())
    B = int(input())
    S = int(input())
    
    final_image = first_assignment(C, fun, Q, N, B, S)()
    test_image = np.load(filename)
    
    #comparing_images(final_image, test_image)
    print("%.4f" % RMSE(final_image, test_image))
