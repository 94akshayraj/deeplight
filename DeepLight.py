import Depth.Depth as D
from Depth.Depth import cv2
cap = cv2.VideoCapture(0)
mode=1
while(1):
	ret, img2=cap.read()
	if mode==1:
    		cv2.namedWindow('video',cv2.WINDOW_NORMAL)
		img2,Ob=D.DepthMap(img2)
		cv2.imshow('video',img2)
    		k=cv2.waitKey(1) & 0xFF
    		if k==27:
        		break
cap.release()
cv2.destroyAllWindows()
