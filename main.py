import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Get Color Image
# # Get Black and white image as 0
# # Get unchanged the row version as -1
# img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
# cv2.imshow("Input Window", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # Change Colour Space
# flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
# print(flags)
#
# # convert to gray
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", imgGray)
#
# # convert to HSV
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# cv2.imshow("HSV Image", imgGray)
#
# cv2.waitKey()
# cv2.destroyAllWindows()
#
# # *************************************************************
# # Lab 2
# # *************************************************************
#
# # Load or read the video
# vid = cv2.VideoCapture("C:/Users/wogza/Downloads/qq.mp4")
#
# # Take each Frame of the video
# while vid.isOpened():
#     ret, frame = vid.read()
#
#     # Converting BGR to HSV color-space
#     hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     # Define lower rang and upper range of blue
#     lower_Blue = np.array([110, 50, 50])
#     upper_Blue = np.array([130, 255, 255])
#
#     # There should be HSV image for a range of blue colour
#     blueMusk = cv2.inRange(hsvImg, lower_Blue, upper_Blue)
#
#     # Extract the blue object alone
#     result = cv2.bitwise_and(frame, frame, mask=blueMusk)
#
#     # Display Original Video
#     originalVideo = cv2.resize(frame, (400, 400))
#     cv2.imshow("Original Video", originalVideo)
#
#     # Display Binary Video
#     binaryVideo = cv2.resize(blueMusk, (400, 400))
#     cv2.imshow("Binary Video", binaryVideo)
#
#     # Display masked / object Video
#     maskedVideo = cv2.resize(result, (400, 400))
#     cv2.imshow("Masked Video", maskedVideo)
#
#     # Condition to break the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# vid.release()
# cv2.destroyAllWindows()

# ******************************************
# LAB 3
# Understanding Histograms

img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Simple Histogram For above image
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

# Simple Histogram For above image OPENCV
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()

# Histogram calculation/ Plotting in NUMPY
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 0)
histr = cv2.calcHist([img], [0], None, [256], [0, 256])
histr, bins = np.histogram(img.ravel(), 256, [0, 256])

# Show graph
plt.plot(hist)
plt.show()

# BGR histograms OPEN CV
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 255])
plt.show()

# BGR histograms Numpy
img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
b, g, r = cv2.split(img)
histr1, bins = np.histogram(b.ravel(), 256, [0, 256])
histr2, bins = np.histogram(g.ravel(), 256, [0, 256])
histr3, bins = np.histogram(r.ravel(), 256, [0, 256])

plt.plot(histr1)
plt.plot(histr2)
plt.plot(histr3)
plt.show()
