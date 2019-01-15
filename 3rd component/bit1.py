import urllib
import cv2
import numpy as np
import time
import pytesseract
from gtts import gTTS
import os

# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://192.168.1.100:8080/shot.jpg'

while True:
    # Use urllib to get the image from the IP camera
    imgResp = urllib.urlopen(url)
    
    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
    # Finally decode the array to OpenCV usable format ;) 
    img = cv2.imdecode(imgNp,-1)
	
	
	# put the image on screen
    cv2.imshow('IPWebcam',img)
    cv2.imwrite('Digits.png',img)
    #To give the processor some less stress
    #time.sleep(0.1) 

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
	#cv2.imwrite('Digits.png',frame)
	cv2.destroyAllWindows()
        break


#READS IMAGE

img = cv2.imread('Digits.png')
#"digits.jpg" is image to be read
#cv2.imshow('image',img)
#show image
#cv2.waitKey(0)
#waits for keystrock

#CHANGE IMAGE TO TEXT

ocr=pytesseract.image_to_string(img)
#read text from image

print ocr
#pints processed text


#TEXT TO SPEECH

# tts = gTTS(ocr, lang='en')
# #input text,lang selectn


# tts.save("audio.mp3")
# #saves to dir where prog is saved


# os.system("play audio.mp3")
# #play saved audio from above step


#---------------------------------------
#pip3 install playsound    #install
#pip3 install gtts         #install    
#sudo apt-get install sox      (for playing mp3 from terminal)
#sudo apt-get install sox libsox-fmt-all
#pip install pytesseract
# pip install opencv-python