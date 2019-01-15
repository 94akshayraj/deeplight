import tensorflow as tf
import sys
import os
import time
import datetime
from utils import *
from pydnet import *
resolution=2
checkpoint_dir='Depth/checkpoint/pydnet'
def Obst(ar):
	if ar[0,0] and ar[0,1] and ar[0,2]:
		return " There is something on Top"
	if ar[1,0] and ar[1,1] and ar[1,2]:
		return " There is Infront of You"
	if ar[2,0] and ar[2,1] and ar[2,2]:
		return " There is something Below you"
	if ar[0,0] and ar[1,0] and ar[2,0]:
		return " There is something on your Right Side"
	if ar[0,2] and ar[1,2] and ar[2,2]:
		return " There is something on your Left Side"
def DepthMap(img):
	with tf.Graph().as_default():
	    	height=512
	    	width=512
	    	obt=int(0.33*512)
	   	tbt=int(0.66*512)
	    	placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}
	    	with tf.variable_scope("model") as scope:
	      		model = pydnet(placeholders)
	    		init = tf.group(tf.global_variables_initializer(),
	               		tf.local_variables_initializer())
	    		loader = tf.train.Saver()
	    		saver = tf.train.Saver()
    		with tf.Session() as sess:
        		sess.run(init)
        		loader.restore(sess,checkpoint_dir)
        	  	img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
        	  	img = np.expand_dims(img, 0)
        	  	start = time.time()
        	  	disp = sess.run(model.results[resolution-1], feed_dict={placeholders['im0']: img})
        		end = time.time()
        	 	disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
	  		disp_color[obt,:,:]=0
          		disp_color[tbt,:,:]=0
	  		disp_color[:,obt,:]=0
	  		disp_color[:,tbt,:]=0
	  		gray = cv2.cvtColor(disp_color, cv2.COLOR_BGR2GRAY)
	  		O=np.zeros(9).reshape(3,3)	
          		O[0,0]=disp_color[0:obt,0:obt,:].mean()
          		O[0,1]=disp_color[0:obt,obt:tbt,:].mean()
          		O[0,2]=disp_color[0:obt,tbt:width,:].mean()
          		O[1,0]=disp_color[obt:tbt,0:obt,:].mean()
          		O[1,1]=disp_color[obt:tbt,obt:tbt,:].mean()
          		O[1,2]=disp_color[obt:tbt,tbt:width,:].mean()
          		O[2,0]=disp_color[tbt:height,0:obt,:].mean()
          		O[2,1]=disp_color[tbt:height,obt:tbt,:].mean()
          		O[2,2]=disp_color[tbt:height,tbt:width,:].mean()
	  		Obstacle_Mat=O>0.5
			Obs=Obst(Obstacle_Mat)
          		del img
          		del disp
			return disp_color,Obs
