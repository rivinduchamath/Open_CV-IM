import cv2

img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg", 1)
cv2.imshow("Input Window", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Change Colour Space
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

# convert to gray
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", imgGray)

# convert to HSV
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", imgGray)

cv2.waitKey()
cv2.destroyAllWindows()

vid = cv2.VideoCapture("C:/Users/wogza/Downloads/q.mp4")
while vid.isOpened():
    ret, frame = vid.read()

    originalVideo = cv2.resize(frame,(400, 400))
    cv2.imshow("Original Video", originalVideo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
