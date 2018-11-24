import numpy as np 
#import system 
import glob
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
import cv2 


def CNN_classifier(input_img):
	num_classes=10 
	conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
	pool1 = MaxPooling2D(pool_size=(3, 3))(conv1) #14 x 14 x 32
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) # 14 x 14 x 64
	pool2 = MaxPooling2D(pool_size=(3, 3))(conv2) #14 x 14 x 32
	flat1 = Flatten()
	dense1=Dense(1000, activation='relu')(flat1)
	classifer=Dense(num_classes, activation='softmax')(dense1)
	return classifer 



### load data ###
train_feature=np.load('feature.npy')
print (np.shape(feature_map))
img_w=np.shape(feature_map)[1]
img_h=np.shape(feature_map)[2]
inChannel=np.shape(feature_map)[3] 
input_img = Input(shape = (x, y, inChannel))

### train setting ###
batch_size = 20
epochs = 120



# batch_size = 20
# epochs = 120
# inChannel = img_channel
# x, y = img_w,img_h
# #print (x,y)
# input_img = Input(shape = (x, y, inChannel))

CNN_classifier = Model(input_img, autoencoder(input_img))
CNN_classifier.compile(loss='mean_squared_error', optimizer = RMSprop())
CNN_classifier.summary()


CNN_classifier_train = CNN_classifier.fit(train_images, train_images, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images, test_images))
#CNN_classifier.save('AE_model.h5')

