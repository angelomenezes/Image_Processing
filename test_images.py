import numpy as np

image = np.array([n+i for i in range(255) for n in range(255)])

print(image)

#plt.imshow(image, cmap="gray")