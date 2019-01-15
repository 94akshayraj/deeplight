import urllib
import cv2
import numpy as np
import time
import pytesseract
from gtts import gTTS
import os

def DocRead(img):    
        
	imgNp = np.array(bytearray(img),dtype=np.uint8)
        
        img = cv2.imdecode(imgNp,-1)

    	ocr=pytesseract.image_to_string(img)

    	return ocr
