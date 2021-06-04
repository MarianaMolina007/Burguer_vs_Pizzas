#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys, time 
import numpy as np
import cv2
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

model = load_model('/home/fabricio/Documentos/embebidos2/Burguer_vs_Pizzas/float_model/f_model_projectBP2.h5')

prev_img=cv2.imread('/home/fabricio/Documentos/embebidos2/Burguer_vs_Pizzas/dataset/test/burger_1.jpg')
curr_img=cv2.imread('/home/fabricio/Documentos/embebidos2/Burguer_vs_Pizzas/dataset/test/burger_6.jpg')


cap = cv2.VideoCapture('/home/fabricio/Documentos/embebidos2/Burguer_vs_Pizzas/videos/test.mp4')
#cap = cv2.VideoCapture(0)
ret, frame = cap.read()
prev_img=frame
prev_img_res=cv2.resize(prev_img,(250,250))
prev_img_c=cv2.cvtColor(prev_img_res, cv2.COLOR_BGR2GRAY)
kernel = np.ones((15,15),np.float32)/225
counter=0
acc=0.0

while(True):
        
	ret, frame = cap.read()
	curr_img_res=cv2.resize(frame,(250,250))
	curr_img_res=curr_img_res/255.0
	#curr_img_c=cv2.cvtColor(curr_img_res, cv2.COLOR_BGR2GRAY)
	#flow = cv2.calcOpticalFlowFarneback(prev_img_c,curr_img_c, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	#vert = cv2.normalize(flow[...,1], None, 0, 255, cv2.NORM_MINMAX)
	#vert = vert.astype('uint8')
	#
	#smoothed = cv2.filter2D(vert,-1,kernel)
#
	#back = cv2.cvtColor(smoothed,cv2.COLOR_GRAY2RGB)

	fall = np.expand_dims(curr_img_res ,axis=0)
	start_time = time.time()
	preds=model.predict(fall)
	acc=acc+(time.time()-start_time)

	if preds[0][0] < 0.5:
		print("PIZZA",preds," FPS: ", 1.0/(time.time()-start_time), counter)
    
	else: 
		print("HAMBURGUER",preds," FPS: ", 1.0/(time.time()-start_time), counter)

	cv2.imshow("frame", frame)
	cv2.waitKey(1)
	prev_img_c=curr_img_res
	#print(preds," FPS: ", 1.0/(time.time()-start_time))
	acc=acc+1.0/(time.time()-start_time)
	counter=counter+1
	#time.sleep(0.05)
	if counter>=1100:
		print('average:',acc/counter)
		break
		
