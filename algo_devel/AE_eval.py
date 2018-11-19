import numpy as np 
#import system 
import glob
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.models import load_model
import cv2 


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


#### evaluation of auto Encoder 

autoEncoder = load_model('models/AE_model.h5') 
recons=autoEncoder.predict(test_images)
print ('recons_shape',np.shape(recons))
print ('recons',recons[0,:,:,:])
cv2.imshow('result',recons[0,:,:,:])
cv2.waitKey()

