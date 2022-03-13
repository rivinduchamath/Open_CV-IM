import cv2

img = cv2.imread("C:/Users/wogza/Pictures/aa.jpg",1)
cv2.imshow("Input Window", img)
cv2.waitKey()
cv2.destroyAllWindows()