import numpy as np
import matplotlib.pyplot as plt

image = np.array([[n for i in range(255)] for n in range(255)])

print(image)

plt.imshow(image, cmap='gray')
plt.show()
