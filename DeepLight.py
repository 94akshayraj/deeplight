import Depth.Depth as D
from Depth.Depth import cv2
#import image_caption.engine as E
#import DocRead.DocRead as DR
cap = cv2.VideoCapture(0)
mode=input("Enter Mode 1/2/3 : ")
while(1):
	ret, img2=cap.read()
	if mode==1:
		print "Obstacle Detection"
		img2,Ob=D.DepthMap(img2)
		print Ob		
		#E.play(Ob)
	if mode==2:
		print "Captioning"
		sent=E.caption(img2)
		#E.play(sent)
		
	if mode==3:
		print "Document Reading"
		doc=DR(img2)
		#E.play(doc)
		
cap.release()
cv2.destroyAllWindows()
