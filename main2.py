import cv2
import numpy as np

# Image negative
img = cv2.imread('"C:/Users/wogza/Downloads/New folder (2)/A/rectangle.png"', 0)

m, n = img.shape

# To find the maximum grey level
# value in the image
L = img.max()

# Maximum grey level value minus
# the original image gives the
# negative image
img_neg = L - img

# convert the np array img_neg to
# a png image
cv2.imwrite('"C:/Users/wogza/Downloads/New folder (2)/A/rectangle.png"', img_neg)

# Thresholding without background
# Let threshold =T
# Let pixel value in the original be denoted by r
# Let pixel value in the new image be denoted by s
# If r<T, s= 0
# If r>T, s=255

T = 150

# create a array of zeros
img_thresh = np.zeros((m, n), dtype=int)

for i in range(m):

    for j in range(n):

        if img[i, j] < T:
            img_thresh[i, j] = 0
        else:
            img_thresh[i, j] = 255

# Convert array to png image
cv2.imwrite('"C:/Users/wogza/Downloads/New folder (2)/A/rectangle.png"', img_thresh)

# the lower threshold value
T1 = 100

# the upper threshold value
T2 = 180

# create a array of zeros
img_thresh_back = np.zeros((m, n), dtype=int)

for i in range(m):

    for j in range(n):

        if T1 < img[i, j] < T2:
            img_thresh_back[i, j] = 255
        else:
            img_thresh_back[i, j] = img[i, j]

# Convert array to png image
cv2.imwrite('"C:/Users/wogza/Downloads/New folder (2)/A/rectangle.png"', img_thresh_back)
