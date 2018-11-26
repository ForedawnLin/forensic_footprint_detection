import numpy as np 
#import system 
import glob
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dense,Flatten
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.losses import mean_squared_error,categorical_crossentropy 
from keras import backend as K

import cv2 

### 1. figure out how to add multiple loss 
### 2. figure out the label file and how to add to data gen kl


def autoencoder_CNN(input_img):
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

	# CNN classifier 
	CNN_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv3) #16 x 16 x 128
	CNN_conv_1 = Conv2D(128, (16, 16), activation='relu')(CNN_pool_1) #1 x 1 x 64
	#CNN_pool_2 = MaxPooling2D(pool_size=(2, 2))(CNN_conv_1) #7 x 7 x 64
	flat1 = Flatten()(CNN_conv_1)
	#dense1=Dense(1000, activation='relu')(flat1)
	classifer=Dense(1175, activation='softmax')(flat1)
	return decoded,classifer



def AE_loss(y_true, y_pred):
	### AE loss 
	y1_pred = y_pred
	y1_true = y_true
	#loss1= K.mean(K.square(y1_true-y1_pred))
	loss1= mean_squared_error(y1_true,y1_pred)
	# print ('y1_pred:',K.int_shape(y1_pred) )
	# print ('y1_true:',K.int_shape(y1_true) )
	# print ('MSE loss1 shape',K.int_shape(loss1))
	return loss1#{'loss1':loss1,'loss2':loss2}

def Classifier_loss(y_true,y_pred):
	y2_pred = y_pred
	y2_true = y_true
	loss2= categorical_crossentropy(y2_true,y2_pred)
	# print ('y2_pred:',K.int_shape(y2_pred) )
	# print ('y2_true:',K.int_shape(y2_true) )
	# print ('CE loss2 shape',K.int_shape(loss2))
	#loss=[loss1,loss2]
	return loss2




### load data ###
mainPath='../../FID-300/tracks_cropped/cropped/train/'  ## main path of the pictures 
pic_fmt='.jpg'  ## picture format 
imagePaths_list=[]  ## init image paths
train_index_path='../data_augmentation/label_train_index.txt' 
train_label_path= '../data_augmentation/label_train.txt'
FileID_train_index=open(train_index_path,'r')
for indice in FileID_train_index:
	indice=indice.replace(' ','')
	index=indice.split(',')
index=index[:-1]
#print (index)
FileID_train_label=open(train_label_path,'r')
for labels in FileID_train_label:
	label=labels.split(',')
label=label[:-1]
#print (label)
imagePaths_list=[mainPath+index[i]+pic_fmt for i in np.arange(len(index))]
label=[int(label[i])-1 for i in np.arange(len(label))]  ## -1 b/c zero index 
max_label=1175

image_num=np.size(imagePaths_list) ### total image numbers 

train_valid_ratio=4
train_num=int(np.floor(image_num/(train_valid_ratio+1)*train_valid_ratio))
valid_num=image_num-train_num

print ('img_num:',image_num)
print ('train_num:',train_num)
print ('valid_num:',valid_num)

train_imagePaths_list=imagePaths_list[:train_num]
train_label=label[:train_num]
valid_imagePaths_list=imagePaths_list[train_num:image_num]
valid_label=label[train_num:image_num]


#### trainig settings ####
train_ord=np.random.permutation(train_num)
train_random_paths=[train_imagePaths_list[i] for i in train_ord] ### randomize training image paths
train_random_label=[train_label[i] for i in train_ord]
#print (train_random_label)


batch_size = 24
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
autoEncoder_CNN = Model(input_img, autoencoder_CNN(input_img)) ### create model 
autoEncoder_CNN.compile(loss=[AE_loss,Classifier_loss], optimizer = 'sgd',loss_weights=[0.5, 0.5])
autoEncoder_CNN.summary()
		




def generate_data(img_paths_list,label_list,total_image_num,batch_size,w,h,max_label):
	### img_paths_list: list contains paths for images 
	### label_list: associated label list 
	### total_image_num: len(paths_list), the total images in the list 
	### batch_size: the batch size 
	### w,h: resize size for input imgage  
	### max_label: the lagest label num, define the one-hot vector dimension 
	i=0 
	while True:
		image_batch=[] ### batched image 
		label_batch=[] ### asscociated batched labels 
		for index in np.arange(batch_size):
			if i==total_image_num:
				i=0 
			img_path=img_paths_list[i]
			label=label_list[i]
			i+=1;			
			img= cv2.imread(img_path)[:,:,1]
			# print ('img',np.shape(img))
			img= cv2.resize(img,(w,h))/255  ### -2 for maxpool and upsample commendation 
			img= np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1)) ## instead of m*n, reshape img to 1*m*n*1 for keras input 
			image_batch.append(img)
			label_vector= np.zeros(max_label)
			# print (np.shape(label_vector))
			label_vector[label]=1
			label_batch.append(label_vector)
		image_batch=np.array(image_batch)
		label_batch=np.array(label_batch)
		#print (np.shape(label_batch))
		yield (image_batch, [image_batch,label_batch]) ##(input, output)



autoEncoder_CNN.fit_generator(generator=generate_data(train_random_paths,train_random_label,train_num,batch_size,resize_w,resize_h,max_label),
                     steps_per_epoch=iters_batch, epochs=epochs,validation_data=generate_data(valid_imagePaths_list,valid_label,valid_num,valid_batch_size,resize_w,resize_h,max_label),validation_steps=valid_iters_batch)


autoEncoder.save('models/AE_CNN_model.h5')

