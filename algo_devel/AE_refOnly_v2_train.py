import numpy as np 
#import system 
import glob
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dense,Flatten,BatchNormalization,Dropout,Deconvolution2D
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.losses import mean_squared_error,categorical_crossentropy 
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import cv2 

### 1. figure out how to add multiple loss 
### 2. figure out the label file and how to add to data gen kl


def autoencoder_CNN(input_img):
	dropRate=0.5

	#drop_input=Dropout(dropRate)(input_img)
	conv1 = Conv2D(256, (3, 3), activation='relu',padding='same')(input_img) #28 x 28 x 32
	#norm1=BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	#drop1=Dropout(dropRate)(pool1)
	

	conv2 = Conv2D(512, (3, 3), activation='relu',padding='same')(pool1) #14 x 14 x 64
	#norm2=BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
	#drop2=Dropout(dropRate)(pool2)
	
	conv3 = Conv2D(16, (3, 3), activation='relu',padding='same')(pool2) #7 x 7 x 128 (small and thick)
	# pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #7 x 7 x 64
	# # #norm3=BatchNormalization()(conv3)

	# conv4 = Conv2D(16, (3, 3), activation='relu')(pool3) #7 x 7 x 128 (small and thick)
	# pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #7 x 7 x 64
		
	# conv5 = Conv2D(2048, (2, 2), activation='relu')(pool4) #7 x 7 x 128 (small and thick)
	# pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #7 x 7 x 64
	

	# #decoder

	# de_conv1 = Deconvolution2D(512, (3, 3), activation='relu',output_shape=(None,1,14,14))(conv4) #7 x 7 x 128
	
	# up1 = UpSampling2D((2,2))(de_conv1)
	de_conv2 = Deconvolution2D(512, (3, 3), activation='relu',output_shape=(None,1,32,32),padding='same')(conv3) #7 x 7 x 128
	

	up2 = UpSampling2D((2,2))(de_conv2)
	de_conv3 = Deconvolution2D(256, (3, 3), activation='relu',output_shape=(None,1,64,64),padding='same')(up2) #7 x 7 x 128
		

	up3 = UpSampling2D((2,2))(de_conv3)
	de_conv4 = Deconvolution2D(1, (3, 3), activation='relu',output_shape=(None,1,128,128),padding='same')(up3) #7 x 7 x 128
	
	# #norm4=BatchNormalization()(conv4)
	# up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
	# #drop4=Dropout(dropRate)(up1)
		
	# conv5 = Deconvolution2D(256, (3, 3), activation='relu',output_shape=(None,1,64,64))(up1) # 14 x 14 x 64
	# #norm5=BatchNormalization()(conv5)
	# up2 = UpSampling2D((4,4))(conv5) # 28 x 28 x 64
	# #drop5=Dropout(dropRate)(up2)
		
	#decoded = Deconvolution2D(1, (3, 3), activation='sigmoid',output_shape=(None,1,130,130))(up2) # 28 x 28 x 1
	decoded =de_conv4


	# CNN classifier 
	# CNN_pool_1 = MaxPooling2D(pool_size=(2, 2))() #16 x 16 x 128
	# # drop6=Dropout(dropRate)(CNN_pool_1)
	
	# CNN_conv_1 = Conv2D(512, (2, 2), activation='relu')(pool2) #1 x 1 x 64
	# # drop7=Dropout(dropRate)(CNN_conv_1)
	# CNN_pool_1 = MaxPooling2D(pool_size=(2, 2))(CNN_conv_1)


	#CNN_pool_2 = MaxPooling2D(pool_size=(2, 2))(CNN_conv_1) #7 x 7 x 64
	# flat1 = Flatten()(conv4)
	# dense1=Dense(2048, activation='relu')(flat1)
	# classifer=Dense(1175, activation='softmax')(dense1)
	return decoded



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
def load_data_test(main_path,index_path,label_path):
	pic_fmt='.jpg'  ## picture format 
	imagePaths_list=[]  ## init image paths
	FileID_index=open(index_path,'r')
	for indice in FileID_index:
		indice=indice.replace(' ','')
		#indice=indice.replace('r','',1)
		index=indice.split(',')
	#print (index)	
	index=index[:-1]
	indice=index 
	# indice=[]
	# for path in index:
	# 	if 'r' in path:
	# 		indice.append('r'+path)
	# 	else:
	# 		indice.append(path)  
	#index=['r'+index[i] for i in np.arange(len(index)) if 'r' in index[i]]  ### name issue, for reference image, its named as r+name_in_file
	#print (indice)
	FileID_label=open(label_path,'r')
	for labels in FileID_label:
		label=labels.split(',')
	label=label[:-1]
	#print (label)
	imagePaths_list=[main_path+indice[i]+pic_fmt for i in np.arange(len(index))]
	label=[int(label[i])-1 for i in np.arange(len(label))]  ## -1 b/c zero index 
	return imagePaths_list,label 



mainPath_train='../../FID-300/tracks_cropped/cropped/train_noise/'  ## main path of the pictures 
train_index_path='../data_augmentation/label_train_index.txt' 
train_label_path= '../data_augmentation/label_train.txt'
imagePaths_list_train,label=load_data_test(mainPath_train,train_index_path,train_label_path)


mainPath_test='../../FID-300/tracks_cropped/cropped/test/'  ## main path of the pictures 
test_index_path='../data_augmentation/label_test_index.txt' 
test_label_path= '../data_augmentation/label_test.txt'
imagePaths_list_test,label_test=load_data_test(mainPath_test,test_index_path,test_label_path)


#print ('train_list:',imagePaths_list_train)
#print (label_test)
print ('label num:',len(list(set(label))))
# print ('train image label',len(label))

max_label=1175

image_num=np.size(imagePaths_list_train) ### total image numbers 

train_num=image_num
#train_valid_ratio=4
#train_num=int(np.floor(image_num/(train_valid_ratio+1)*train_valid_ratio))
#valid_num=image_num-train_num


train_imagePaths_list=imagePaths_list_train[:train_num]
train_label=label[:train_num]
#valid_imagePaths_list=imagePaths_list[train_num:image_num]
#valid_label=label[train_num:image_num]
valid_imagePaths_list=imagePaths_list_test
valid_label=label_test
valid_num=len(valid_label)
print ('valid_image_num', len(valid_imagePaths_list))
print ('valid_label_num',valid_num)

print ('img_num:',image_num)
print ('train_num:',train_num)
# print ('valid_num:',valid_num)
print ('test_num:',len(valid_label))
# print('test_img_path_num',valid_imagePaths_list)



#### trainig settings ####
train_ord=np.random.permutation(train_num)
train_random_paths=[train_imagePaths_list[i] for i in train_ord] ### randomize training image paths
train_random_label=[train_label[i] for i in train_ord]
#print (len(train_random_label))




batch_size = 32
iters_batch = int(np.floor(np.true_divide(train_num,batch_size)))
epochs = 47

#n_valid_check=50 ### number of validation images for check at each iteration  
#valid_batch_size = 50
valid_batch_size = len(label_test)/10
valid_iters_batch = int(np.floor(np.true_divide(valid_num,valid_batch_size)))




n_imgs=batch_size ### input layer image number 
print(train_imagePaths_list[0])
img= cv2.imread(train_imagePaths_list[0])[:,:,1]
#cv2.imshow('img',img)
#cv2.waitKey()
img_h=np.shape(img)[0] ### input layer image height 
img_w=np.shape(img)[1] ### input layer image width
img_channel=1 ### input layer image width, gray image 	
print ('imag_shape',img_h,img_w,img_channel)

resize_w=128; ### resize image to before feeding into network 
resize_h=128;
input_img = Input(shape = (resize_w, resize_h, img_channel)) ### -2 for maxpool and upsample commendation 
autoEncoder_CNN = Model(input_img, autoencoder_CNN(input_img)) ### create model 
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


# for img_path in train_random_paths:
# 	img_list=img_path.split('/')
# 	img_path_no_noise=img_list[0]+'/'+img_list[1]+'/'+img_list[2]+'/'+img_list[3]+'/'+img_list[4]+'/'+'train'+'/'+img_list[6]
# 	#print (img_path_no_noise)
# 	img_noNoise=[]

# 	img_noNoise= cv2.imread(img_path_no_noise)[:,:,1]



autoEncoder_CNN.compile(loss='mean_squared_error',optimizer = 'sgd')
print ('metric_name:',autoEncoder_CNN.metrics_names)
autoEncoder_CNN.summary()
		







def generate_data(img_paths_list,label_list,total_image_num,batch_size,w,h,max_label,train_set):
	### img_paths_list: list contains paths for images 
	### label_list: associated label list 
	### total_image_num: len(paths_list), the total images in the list 
	### batch_size: the batch size 
	### w,h: resize size for input imgage  
	### max_label: the lagest label num, define the one-hot vector dimension 
	i=0 
	while True:
		image_batch=[] ### batched image 
		image_noNoise_batch=[] ### no noise 
		#label_batch=[] ### asscociated batched labels 
		for index in np.arange(batch_size):
			if i==total_image_num:
				i=0 
				train_ord=np.random.permutation(total_image_num)
				img_paths_list=[img_paths_list[j] for j in train_ord] ### randomize training image paths
			img_path=img_paths_list[i]
			label=label_list[i]
			i+=1;	
			img= cv2.imread(img_path)[:,:,1]
			# print ('img',np.shape(img))
			img= cv2.resize(img,(w,h))/255  ### -2 for maxpool and upsample commendation 
			img= np.reshape(img,(np.shape(img)[0],np.shape(img)[1],1)) ## instead of m*n, reshape img to 1*m*n*1 for keras input 
			image_batch.append(img)
			
			if train_set==1:
				img_list=img_path.split('/')
				img_path_no_noise=img_list[0]+'/'+img_list[1]+'/'+img_list[2]+'/'+img_list[3]+'/'+img_list[4]+'/'+'train'+'/'+img_list[6]
				#print (img_path_no_noise)
				img_noNoise= cv2.imread(img_path_no_noise)[:,:,1]
				# print ('img',np.shape(img))
				img_noNoise= cv2.resize(img_noNoise,(w,h))/255  ### -2 for maxpool and upsample commendation 
				img_noNoise= np.reshape(img_noNoise,(np.shape(img_noNoise)[0],np.shape(img_noNoise)[1],1)) ## instead of m*n, reshape img to 1*m*n*1 for keras input 
				image_noNoise_batch.append(img_noNoise)


			#label_vector= np.zeros(max_label)
			# print (np.shape(label_vector))
			#label_vector[label]=1
			#label_batch.append(label_vector)
		image_batch_input=np.array(image_batch)
		image_batch_output=[]
		if train_set==1:
			image_batch_output=np.array(image_noNoise_batch)
		else:
			image_batch_output=np.array(image_batch)
		#label_batch=np.array(label_batch)
		#print (np.shape(label_batch))
		yield (image_batch_input, image_batch_output) ##(input, output)


### load model ###
autoEncoder_CNN.load_weights('models/results/AE_refOnly_v2_weights.03-0.01.hdf5')


# # checkpoint
filepath="models/AE_refOnly_v2_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False) ### save model based on classification loss 
callbacks_list = [checkpoint]


MODEL=autoEncoder_CNN.fit_generator(generator=generate_data(train_random_paths,train_random_label,train_num,batch_size,resize_w,resize_h,max_label,1),
                     steps_per_epoch=iters_batch, epochs=epochs,validation_data=generate_data(valid_imagePaths_list,valid_label,valid_num,valid_batch_size,resize_w,resize_h,max_label,0),validation_steps=valid_iters_batch,callbacks=callbacks_list,shuffle=True)


val_loss=MODEL.history['val_loss']


train_loss=MODEL.history['loss']

loss_file=open('models/loss_ref_v2_4_51.txt','a')
for i in np.arange(len(val_loss)): 
	loss_file.write(str(val_loss[i])+' '+str(train_loss[i])+'\n')







autoEncoder_CNN.save('models/AE_refOnly_v2.h5')
autoEncoder_CNN.save_weights("models/AE_refOnly_v2_weights.hdf5")

