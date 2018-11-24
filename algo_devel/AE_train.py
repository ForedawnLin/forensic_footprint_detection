import numpy as np 
#import system 
import glob
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
import cv2 


def autoencoder(input_img):
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)

	#decoder
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 128
	up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
	conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 64
	up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
	return decoded




# img = np.random.rand(65,65,1)
# input_img_path='C:/Users/tongl/Desktop/CMU course/10701/project/FID-300/tracks_cropped/cropped/1_2.jpg'
# img = image.load_img(input_img_path, target_size=(65, 65))
# print(type(img))
# input_img = image.img_to_array(img)
# input_img=input_img[:,:,1]
# print (x)
# io.imshow(x)
# io.show()
# cv2.imshow("1",input_img)
# cv2.waitKey()
### load data ###

images = []
for imgPath in glob.glob("../../FID-300/cropped/*.jpg"):#tracks_cropped/cropped/train/*.jpg"):
    print (imgPath)
    img= cv2.imread(imgPath)[:,:,1]
    img=cv2.resize(img,(np.shape(img)[0]-1,np.shape(img)[1]-1))/255
    img= np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1)) ## instead of m*n, reshape img to m*n*1 for keras input  
    #print (np.shape(img))
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

# img = np.random.rand(65,65,1)
# input_img_path='C:/Users/tongl/Desktop/CMU course/10701/project/FID-300/tracks_cropped/cropped/1_2.jpg'
# img = cv2.imread(input_img_path)[:,:,1]
# print (np.shape(img))

# cv2.imshow('img',images[0])
# cv2.waitKey()

batch_size = 20
epochs = 120
inChannel = img_channel
x, y = img_w,img_h
#print (x,y)
input_img = Input(shape = (x, y, inChannel))

autoEncoder = Model(input_img, autoencoder(input_img))
autoEncoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoEncoder.summary()


autoencoder_train = autoEncoder.fit(train_images, train_images, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images, test_images))
autoEncoder.save('models/AE_model.h5')

