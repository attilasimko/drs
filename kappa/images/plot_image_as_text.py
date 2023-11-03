from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open("smiley.png").convert('L')
arr = np.array(img.getdata())
arr = np.reshape(arr, (8, 8))
plt.imshow(arr, cmap='gray')
plt.show()