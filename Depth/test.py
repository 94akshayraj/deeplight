import Depth
from Depth import cv2
img = cv2.imread('img.jpg')
img1,Ob=Depth.DepthMap(img)
cv2.imshow('image',img1)
cv2.waitKey(0)
