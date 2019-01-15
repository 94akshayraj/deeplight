import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *

resolution=2
checkpoint_dir='checkpoint/IROS18/pydnet'
def main(_):

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
    cam = cv2.VideoCapture(0)
    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess,checkpoint_dir)
        while True:
          for i in range(4):
            cam.grab()
          ret_val, img = cam.read()
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img = np.expand_dims(img, 0)
          start = time.time()
          disp = sess.run(model.results[resolution-1], feed_dict={placeholders['im0']: img})
          end = time.time()
          disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
          #toShow = (np.concatenate((img, disp_color), 0)*255.).astype(np.uint8)
          #toShow = cv2.resize(toShow, (width/2, height))
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
	  Ot=O>0.5
	  print Ot
          cv2.imshow('Deep_Light', disp_color)
          k = cv2.waitKey(1)
          if k == 1048603 or k == 27:
            break
          if k == 1048688:
            cv2.waitKey(0)
          print("Time: " + str(end - start))
          del img
          del disp
          #del toShow
        cam.release()

if __name__ == '__main__':
    tf.app.run()

