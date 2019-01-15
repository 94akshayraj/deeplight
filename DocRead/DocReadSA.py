import urllib
import cv2
import numpy as np
import time
import pytesseract
from gtts import gTTS
import os

def DocRead:
    url='http://192.168.1.100:8080/shot.jpg'

    while True:
        
        imgResp = urllib.urlopen(url)
        
        
        imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
        
        
        img = cv2.imdecode(imgNp,-1)
    	
    	
    	
        cv2.imshow('IPWebcam',img)
        cv2.imwrite('Digits.png',img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
    	cv2.destroyAllWindows()
            break


    #READS IMAGE

    img = cv2.imread('Digits.png')

    ocr=pytesseract.image_to_string(img)

    print ocr

