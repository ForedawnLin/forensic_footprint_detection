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



### load data ###

imagePaths_list=[]
for imgPath in glob.glob("../../FID-300/tracks_cropped/cropped/train/*.jpg"):
	    #print (imgPath)
    imagePaths_list.append(imgPath)
image_num=np.size(imagePaths_list) ### total image numbers 

train_valid_ratio=4
train_num=int(np.floor(image_num/(train_valid_ratio+1)*train_valid_ratio))
valid_num=image_num-train_num

print ('img_num:',image_num)
print ('train_num:',train_num)
print ('valid_num:',valid_num)

train_imagePaths_list=imagePaths_list[:train_num]
valid_imagePaths_list=imagePaths_list[train_num:image_num]



#### trainig settings ####
train_ord=np.random.permutation(train_num)
train_random_paths=[train_imagePaths_list[i] for i in train_ord] ### randomize training image paths


batch_size = 20
iters_batch = int(np.floor(np.true_divide(train_num,batch_size)))
epochs = 10

#n_valid_check=50 ### number of validation images for check at each iteration  
valid_batch_size = 50
valid_iters_batch = int(np.floor(np.true_divide(valid_num,valid_batch_size)))




n_imgs=batch_size ### input layer image number 
img= cv2.imread(imagePaths_list[0])[:,:,1]
img_h=np.shape(img)[0] ### input layer image height 
img_w=np.shape(img)[1] ### input layer image width
img_channel=1 ### input layer image width, gray image 	
print ('imag_shape',img_h,img_w,img_channel)

resize_w=128; ### resize image to before feeding into network 
resize_h=128;
input_img = Input(shape = (resize_w, resize_h, img_channel)) ### -2 for maxpool and upsample commendation 
autoEncoder = Model(input_img, autoencoder(input_img)) ### create model 
autoEncoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoEncoder.summary()
		




# for iteration in np.arange(epochs):
# 	for iter_batch in np.arange(iters_batch): ### ignore the last few batches 
# 		train_images=[]  ### initilize train images 
# 		#print ('iter_batch',(iter_batch+1)*batch_size)
# 		for train_imgPath in train_random_paths[(iter_batch)*batch_size:(iter_batch+1)*batch_size]:
# 			img= cv2.imread(train_imgPath)[:,:,1]
# 			#print ('img',np.shape(img))
# 			img=cv2.resize(img,(np.shape(img)[0]-2,np.shape(img)[1]-2))/255  ### -2 for maxpool and upsample commendation 
# 			img= np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1)) ## instead of m*n, reshape img to m*n*1 for keras input 
# 			train_images.append(img)

# 		#print ('train_images',np.shape(train_images))   
# 		print ('iter_batch',iter_batch)	
# 		train_images=[train_images]
# 		#autoencoder_train = autoEncoder.fit(train_images, train_images, batch_size=20,epochs=1,verbose=1)
# 		autoencoder_train = autoEncoder.fit(train_images, train_images, batch_size=20,epochs=1,verbose=1)



def generate_data(paths_list,total_image_num,batch_size,w,h):
	### paths_list: list contains paths for images 
	### total_image_num: len(paths_list), the total images in the list 
	### batch_size: the batch size 
	### w,h: resize size for input imgage  
	i=0 
	while True:
		image_batch=[]
		for index in np.arange(batch_size):
			if i==total_image_num:
				i=0 
			img_path=paths_list[i]
			i+=1;			
			img= cv2.imread(img_path)[:,:,1]
			#print ('img',np.shape(img))
			img= cv2.resize(img,(w,h))/255  ### -2 for maxpool and upsample commendation 
			img= np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1)) ## instead of m*n, reshape img to 1*m*n*1 for keras input 
			image_batch.append(img)
		image_batch=np.array(image_batch)
		yield (image_batch, image_batch)



autoEncoder.fit_generator(generator=generate_data(train_random_paths,train_num,batch_size,resize_w,resize_h),
                    steps_per_epoch=iters_batch, epochs=epochs,validation_data=generate_data(valid_imagePaths_list,valid_num,valid_batch_size,resize_w,resize_h),validation_steps=valid_iters_batch)



# 	### validation check ####
# 	valid_ord=np.random.permutation(valid_num)
# 	valid_ord=valid_ord[:n_valid_check] ### only get n_valid_check images 
# 	valid_random_paths=[valid_imagePaths_list[i] for i in valid_ord]
# 	valid_images=[] ### initialize valid images 
# 	for valid_image_path in valid_random_paths:
# 			img= cv2.imread(valid_image_path)[:,:,1]
# 			img=cv2.resize(img,(np.shape(img)[0]-1,np.shape(img)[1]-1))/255
# 			img= np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1)) ## instead of m*n, reshape img to m*n*1 for keras input 
# 			valid_images.append(img)
# 	autoEncoder.evaluate(x=valid_images,y=valid_images)
		  	



#### split into train test ####
# n_imgs=np.shape(images)[0]
# img_h=np.shape(images)[1]
# img_w=np.shape(images)[2]
# img_channel=np.shape(images)[3]
# train_test_ratio=4
# train_num=int(np.floor(n_imgs/(train_test_ratio+1)*train_test_ratio))
# test_num=n_imgs-train_num

# train_images=np.array(images[:train_num])
# test_images=np.array(images[train_num:])
# print (np.shape(test_images))

# inChannel = img_channel
# x, y = img_w,img_h
# input_img = Input(shape = (x, y, inChannel))

# autoEncoder = Model(input_img, autoencoder(input_img))
# autoEncoder.compile(loss='mean_squared_error', optimizer = RMSprop())
# autoEncoder.summary()

# autoencoder_train = autoEncoder.fit(train_images, train_images, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_images, test_images))

#autoEncoder.save('models/AE_model.h5')

