import cv2
import numpy as np

# Load Image
img = cv2.imread("C:/Users/wogza/Downloads/New folder (2)/A/rectangle.png", 0)

# Find width and height of image
row = img.shape[0]
column = img.shape[1]

# Create an zeros array
image = np.zeros((row, column), dtype=np.uint8)

# min and max range
min_range = 70
max_range = 90

for i in range(row):
    for j in range(column):

        if min_range < img[i, j] < max_range:
            image[i, j] = 255
        else:
            image[i, j] = img[i, j]

result = np.hstack((img, image))
cv2.imshow("Input and output Image", result)
cv2.waitKey(0)
