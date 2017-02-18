import cv2 as cv

img = cv.imread('path')
img = cv.resize(img, (800,800))
cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

cv.imwrite("img0101.bmp",img)
cv.waitKey(0)
cv.destroyAllWindows()
