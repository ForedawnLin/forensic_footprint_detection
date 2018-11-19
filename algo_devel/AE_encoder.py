import numpy as np 
#import system 
import glob
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import load_model
import cv2 




def get_encoder(AE_model_path,input_img):
	### AE_model_path: AE model file path 
	### input_img: keras input layer
	### return: encoder structure (based on the input AE model) with trained weights  
	autoEncoder = load_model(AE_model_path) 
	layers_to_get=int((np.shape(autoEncoder.layers)[0]-1)/2) ### AE model always have odd layers (even AE layers + one input layer)
	layers1=autoEncoder.layers[1] ## get input layer
	layers=layers1(input_img) ### initilize first layer 
	for layers_to_add in autoEncoder.layers[2:layers_to_get+1]:
		#print (layers_to_add)
		layers=layers_to_add(layers)
	return layers







#### load data(temp) ####
images = []
for imgPath in glob.glob("../../FID-300/tracks_cropped/cropped/*.jpg"):
    img= cv2.imread(imgPath)[:,:,1]
    img=cv2.resize(img,(np.shape(img)[0]-1,np.shape(img)[1]-1))/255
    img= np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1)) ## instead of m*n, reshape img to m*n*1 for keras input  
    images.append(img)
    if np.shape(images)[0]==100:
    	break 

print (np.shape(images))

#### split into train test ####
n_imgs=np.shape(images)[0]
img_h=np.shape(images)[1]
img_w=np.shape(images)[2]
img_channel=np.shape(images)[3]
train_test_ratio=4
train_num=int(np.floor(n_imgs/(train_test_ratio+1)*train_test_ratio))
test_num=n_imgs-train_num

train_images=np.array(images[:train_num])
test_images=np.array(images[train_num:])
print (np.shape(test_images))





inChannel = img_channel
x, y = img_w,img_h
#print (x,y)
input_img = Input(shape = (x, y, inChannel))

########## load data finished ########



### load encoder part of AE ###
AE_model_path='models/AE_model.h5'
encoder = Model(input_img,get_encoder(AE_model_path,input_img))
encoder.summary()



### get extracted features ### 
features=encoder.predict(test_images)
print (np.shape(features))
feature=features[0,:,:,0] ### get one feature 
feature=np.divide(feature-np.min(feature),np.max(feature)-np.min(feature)) ### normalize features 

print (np.shape(feature))
print ('one feature',feature)
cv2.imshow('result',feature) ## show one feature 
cv2.waitKey()
