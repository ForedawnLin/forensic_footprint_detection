import numpy as np 
#import system 
import glob
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dense,Flatten,BatchNormalization,Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.preprocessing import image
from keras.losses import mean_squared_error,categorical_crossentropy 
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import cv2 

### 1. figure out how to add multiple loss 
### 2. figure out the label file and how to add to data gen kl


def autoencoder_CNN(input_img):
	dropRate=0.1

	drop_input=Dropout(0.1)(input_img)
	conv_init = Conv2D(64,(3, 3), activation='relu',padding='same')(drop_input) #28 x 28 x 32
	#norm1=BatchNormalization()(conv1)
	conv1 = Conv2D(128, (3, 3), activation='relu')(conv_init) #28 x 28 x 32
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	#drop1=Dropout(dropRate)(pool1)
	

	conv2 = Conv2D(256, (3, 3), activation='relu')(pool1) #14 x 14 x 64
	#norm2=BatchNormalization()(conv2)
	conv2_2 = Conv2D(512, (3, 3), activation='relu')(conv2) #14 x 14 x 64
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2) #7 x 7 x 64
	#drop2=Dropout(dropRate)(pool2)
	
	conv3 = Conv2D(1024, (3, 3), activation='relu')(pool2) #7 x 7 x 128 (small and thick)
	conv3_3 = Conv2D(2048, (3, 3), activation='relu')(conv3) #7 x 7 x 128 (small and thick)
	# norm3=BatchNormalization()(conv3)
	
	

	# CNN classifier 
	CNN_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv3_3) #16 x 16 x 128
	#drop6=Dropout(dropRate)(CNN_pool_1)
	
	conv4 = Conv2D(4096, (3, 3), activation='relu')(CNN_pool_1) #7 x 7 x 128 (small and thick)
	#conv3_3 = Conv2D(2048, (3, 3), activation='relu')(conv3) #7 x 7 x 128 (small and thick)
	
	# CNN_conv_2 = Conv2D(4096, (3, 3), activation='relu')(drop6) #1 x 1 x 64
	# #drop7=Dropout(dropRate)(CNN_conv_1)
	# CNN_pool_2 = MaxPooling2D(pool_size=(2, 2))(CNN_conv_2) #16 x 16 x 128
	# drop7=Dropout(dropRate)(CNN_pool_2)
	

	# CNN_conv_3 = Conv2D(2048, (3, 3), activation='relu')(drop7) #1 x 1 x 64
	# #drop7=Dropout(dropRate)(CNN_conv_1)
	# CNN_pool_3 = MaxPooling2D(pool_size=(2, 2))(CNN_conv_3) #16 x 16 x 128
	# drop8=Dropout(dropRate)(CNN_pool_3)
	
	# CNN_conv_4 = Conv2D(4096, (3, 3), activation='relu')(drop8) #1 x 1 x 64
	# #drop7=Dropout(dropRate)(CNN_conv_1)
	# CNN_pool_4 = MaxPooling2D(pool_size=(2, 2))(CNN_conv_4) #16 x 16 x 128
	# drop9=Dropout(dropRate)(CNN_pool_4)
	
	#CNN_pool_5 = MaxPooling2D(pool_size=(2, 2))(drop7) #16 x 16 x 128
	#CNN_conv_5 = Conv2D(130, (10,10), activation='relu')(conv4) #1 x 1 x 1175
	
	# #CNN_pool_2 = MaxPooling2D(pool_size=(2, 2))(CNN_conv_1) #7 x 7 x 64
	flat1 = Flatten()(conv4)
	dense1=Dense(1000, activation='relu')(flat1)
	classifer=Dense(130, activation='softmax')(flat1)
	#classifer=flat1
	return classifer



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
def load_data(main_path,index_path,label_path):
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

mainPath_train='../../FID-300/tracks_cropped/cropped/train/'  ## main path of the pictures 
train_index_path='../data_augmentation/label_train_index.txt' 
train_label_path= '../data_augmentation/label_train.txt'
imagePaths_list_train,label=load_data(mainPath_train,train_index_path,train_label_path)

mainPath_test='../../FID-300/tracks_cropped/cropped/test/'  ## main path of the pictures 
test_index_path='../data_augmentation/label_test_index.txt' 
test_label_path= '../data_augmentation/label_test.txt'
imagePaths_list_test,label_test=load_data(mainPath_test,test_index_path,test_label_path)


#print ('train_list:',imagePaths_list_train)
print (label_test)
print ('label num:',len(list(set(label))))
# print ('train_label_set',set(label))
# print ('test_label_set',set(label_test))
# print ('set_diff',set(label)-set(label_test))




def process_label(test_label_set,train_label_set):
	### the function process train label and test labe so that the labels are from 1:130 
	### new labels are 1 based 
	dic_classes={} 
	i=0
	test_label_set_new=[]
	reference_table={} ### look up table for new labels 
	for classes in set(train_label_set):
		reference_table[classes]=i
		i+=1
	#print ('reference_table',reference_table) 
	# for label in train_label_set:
	# 	try:
	# 		a=1
	# 		#print (reference_table[label]) 
	# 	except:
	# 		a=1 
	# 		#print ('key error',label)

	train_label_set_new=[reference_table[label] for label in train_label_set]
	test_label_set_new=[reference_table[label] for label in test_label_set]

	return train_label_set_new,test_label_set_new,reference_table 	

label,label_test,reference_table=process_label(label_test,label)

# print (label_test)
# print ('label num:',len(list(set(label))))
# print('reference_table',reference_table)


max_label=130


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
epochs = 1000	

#n_valid_check=50 ### number of validation images for check at each iteration  
#valid_batch_size = 50
valid_batch_size = len(label_test)
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
autoEncoder_CNN.compile(loss=[Classifier_loss],optimizer = 'sgd')
print ('metric_name:',autoEncoder_CNN.metrics_names)
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
				train_ord=np.random.permutation(total_image_num)
				img_paths_list=[img_paths_list[i] for i in train_ord] ### randomize training image paths
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
		yield (image_batch, label_batch) ##(input, output)


### load model 
#autoEncoder_CNN.load_weights('models/CNN_130_model_weights.27-6.40.hdf5')

# checkpoint
filepath="models/AE_CNN_130_model_weights_conti.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',save_weights_only=False) ### save model based on classification loss 
callbacks_list = [checkpoint]


autoEncoder_CNN.fit_generator (generator=generate_data(train_random_paths,train_random_label,train_num,batch_size,resize_w,resize_h,max_label),
                     steps_per_epoch=iters_batch, epochs=epochs,validation_data=generate_data(valid_imagePaths_list,valid_label,valid_num,valid_batch_size,resize_w,resize_h,max_label),validation_steps=valid_iters_batch,callbacks=callbacks_list,shuffle=True)


autoEncoder_CNN.save('models/AE_CNN_130_model.h5')
autoEncoder_CNN.save_weights("models/AE_CNN_130_model_weights.h5")

