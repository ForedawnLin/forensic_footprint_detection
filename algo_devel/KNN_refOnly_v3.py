from sklearn.neighbors import KNeighborsClassifier


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
from tempfile import TemporaryFile

def autoencoder_CNN(input_img):
	dropRate=0.5

	#drop_input=Dropout(dropRate)(input_img)
	conv1 = Conv2D(64, (3, 3), activation='relu',padding='same')(input_img) #28 x 28 x 32
	#norm1=BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	#drop1=Dropout(dropRate)(pool1)
	

	conv2 = Conv2D(128, (3, 3), activation='relu',padding='same')(pool1) #14 x 14 x 64
	#norm2=BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
	#drop2=Dropout(dropRate)(pool2)
	
	conv3 = Conv2D(256, (3, 3), activation='relu',padding='same')(pool2) #7 x 7 x 128 (small and thick)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #7 x 7 x 64
	# # #norm3=BatchNormalization()(conv3)

	conv4 = Conv2D(512, (3, 3), activation='relu',padding='same')(pool3) #7 x 7 x 128 (small and thick)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #7 x 7 x 64
		
	conv5 = Conv2D(1024, (3, 3), activation='relu',padding='same')(pool4) #7 x 7 x 128 (small and thick)
	pool5 = MaxPooling2D(pool_size=(2, 2))(conv5) #7 x 7 x 64
	
	conv6 = Conv2D(4096, (4, 4), activation='relu')(pool5) #7 x 7 x 128 (small and thick)

	# #decoder

	# de_conv1 = Deconvolution2D(512, (3, 3), activation='relu',output_shape=(None,1,14,14))(conv4) #7 x 7 x 128
	
	# up1 = UpSampling2D((2,2))(de_conv1)
	de_conv1 = Deconvolution2D(1024, (4, 4), activation='relu',output_shape=(None,1,4,4))(conv6) #7 x 7 x 128
	

	up2 = UpSampling2D((2,2))(de_conv1)
	de_conv3 = Conv2D(512, (3, 3), activation='relu',padding='same')(up2) #7 x 7 x 128
		

	up3 = UpSampling2D((2,2))(de_conv3)
	de_conv4 = Conv2D(256, (3, 3), activation='relu',padding='same')(up3) #7 x 7 x 128
	
	up4 = UpSampling2D((2,2))(de_conv4)
	de_conv5 = Conv2D(128, (3, 3), activation='relu',padding='same')(up4) #7 x 7 x 128
	
	up5 = UpSampling2D((2,2))(de_conv5)
	de_conv6 = Conv2D(64, (3, 3), activation='relu',padding='same')(up5) #7 x 7 x 128
		
	up6 = UpSampling2D((2,2))(de_conv6)
	de_conv7 = Conv2D(1, (3, 3), activation='relu',padding='same')(up6) #7 x 7 x 128
	
	decoded =de_conv7

	return decoded




def encoder(input_img):
	
	conv1 = Conv2D(64, (3, 3), activation='relu',padding='same')(input_img) #28 x 28 x 32
	#norm1=BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
	conv2 = Conv2D(128, (3, 3), activation='relu',padding='same')(pool1) #14 x 14 x 64
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
	conv3 = Conv2D(256, (3, 3), activation='relu',padding='same')(pool2) #7 x 7 x 128 (small and thick)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #7 x 7 x 64
	conv4 = Conv2D(512, (3, 3), activation='relu',padding='same')(pool3) #7 x 7 x 128 (small and thick)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) #7 x 7 x 64		
	conv5 = Conv2D(1024, (3, 3), activation='relu',padding='same')(pool4) #7 x 7 x 128 (small and thick)
	pool5 = MaxPooling2D(pool_size=(2, 2))(conv5) #7 x 7 x 64
	conv6 = Conv2D(4096, (4, 4), activation='relu')(pool5) #7 x 7 x 128 (small and thick)
	encoded =conv6
	return encoded



def AE_loss(y_true, y_pred):
	### AE loss 
	y1_pred = y_pred
	y1_true = y_true
	loss1= mean_squared_error(y1_true,y1_pred)
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
	index=index[:-1]
	indice=index
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

# mainPath_train='../../FID-300/tracks_cropped/cropped/reference/'  ## main path of the pictures 
# train_index_path='../data_augmentation/label_reference_index.txt' 
# train_label_path= '../data_augmentation/label_reference.txt'
# imagePaths_list_train,label=load_data(mainPath_train,train_index_path,train_label_path)



mainPath_train='../../FID-300/tracks_cropped/cropped/train/'  ## main path of the pictures 
train_index_path='../data_augmentation/label_train_index.txt' 
train_label_path= '../data_augmentation/label_train.txt'
imagePaths_list_train,label=load_data(mainPath_train,train_index_path,train_label_path)


mainPath_test='../../FID-300/tracks_cropped/cropped/test/'  ## main path of the pictures 
test_index_path='../data_augmentation/label_test_index.txt' 
test_label_path= '../data_augmentation/label_test.txt'
imagePaths_list_test,label_test=load_data(mainPath_test,test_index_path,test_label_path)

print (imagePaths_list_test)
print ('label_test:',label_test)
print ('label num:',len(list(set(label))))




max_label=1175

image_num=np.size(imagePaths_list_train) ### total image numbers 

train_num=image_num


train_imagePaths_list=imagePaths_list_train[:train_num]
train_label=label[:train_num]

valid_imagePaths_list=imagePaths_list_test
valid_label=label_test
valid_num=len(valid_label)
#print ('test_label',label_test)
print ('valid_image_num', len(valid_imagePaths_list))
print ('valid_label_num',valid_num)

print ('img_num:',image_num)
print ('train_num:',train_num)
#print ('valid_num:',valid_num)
print ('test_num:',len(valid_label))



#### trainig settings ####
train_ord=np.random.permutation(train_num)
train_random_paths=[train_imagePaths_list[i] for i in train_ord] ### randomize training image paths
train_random_label=[train_label[i] for i in train_ord]
#print (train_random_label)


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
autoEncoder_CNN.compile(loss=[AE_loss],optimizer = 'Adagrad',loss_weights=[0.5])
print ('metric_name:',autoEncoder_CNN.metrics_names)
#autoEncoder_CNN.summary()


def generate_data(img_paths_list,label_list,total_image_num,batch_size,w,h,max_label):
	### img_paths_list: list contains paths for images 
	### label_list: associated label list 
	### total_image_num: len(paths_list), the total images in the list 
	### batch_size: the batch size 
	### w,h: resize size for input imgage  
	### max_label: the lagest label num, define the one-hot vector dimension 
	image_batch=[] ### batched image 
	label_batch=[] ### asscociated batched labels 
	i=0
	for index in np.arange(batch_size):
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
		#label_vector[label]=1
		label_batch.append(label)
	image_batch=np.array(image_batch)
	label_batch=np.array(label_batch)
	#print (np.shape(label_batch))
	return image_batch, label_batch ##(input, output)




#### Evaluation #######
		
# autoEncoder_CNN.load_weights('models/results/AE_refOnly_v2_weights.03-0.01.hdf5')


### test eval #######

######## label check ##########
# data_x,_=generate_data(valid_imagePaths_list,valid_label,valid_num,valid_batch_size,resize_w,resize_h,max_label) 
# img_pred=autoEncoder_CNN.predict(data_x, batch_size=None, verbose=0, steps=None)

####### reconstruction check #########
# for i in (14,15):
# 	print (valid_imagePaths_list[i])
# 	img2check=i ### the image in test set to check 
# 	img_orig= cv2.imread(valid_imagePaths_list[img2check])[:,:,1]
# 	img_orig= cv2.resize(img_orig,(128,128))/255  ### -2 for maxpool and upsample commendation 
# 	img_pred_test=img_pred[img2check,:,:,0]
# 	img_pred_test=np.squeeze(img_pred_test)
# 	cv2.imshow('orig',img_orig)
# 	cv2.imshow('recons',img_pred_test)
# 	cv2.waitKey()


############# KNN ##################

input_img2 = Input(shape = (resize_w, resize_h, img_channel)) ### -2 for maxpool and upsample commendation 
Encoder = Model(input_img2, encoder(input_img2)) ### create model 
Encoder.load_weights('models/AE_refOnly_v3_weights.38-0.00.hdf5',by_name=True)


feature_Xs=[]
label_Xs=[]

for i in np.arange(100):
	data_X,label_X=generate_data(train_random_paths[i*1175:(i+1)*1175],train_random_label[i*1175:(i+1)*1175],1175,1175,resize_w,resize_h,max_label) 
	#print ('shape',np.shape(data_X))
	feature_X=Encoder.predict(data_X, batch_size=None, verbose=0, steps=None)
	feature_X=np.squeeze(feature_X)
	print (i)
	print ('shape',np.shape(feature_X))
	if i==0:
		feature_Xs=feature_X
		label_Xs=label_X 
	else :
		feature_Xs=np.concatenate((feature_Xs,feature_X),axis=0)
		label_Xs=np.concatenate((label_Xs,label_X),axis=0)
	print ('shape',np.shape(feature_Xs))
	print ('shape',np.shape(label_Xs))

#print ("train_feature_shape:",np.shape(feature_X))
# print ("label_X_shape:",np.shape(label_X))

# feature_X=feature_X[:,:,:,5:15]
print ("train_feature_shape:",np.shape(feature_X))


np.save('featureXs', feature_Xs)
np.save('labelX', label_Xs)


X=[feature_Xs[i,:].flatten('C') for i in np.arange(np.shape(feature_Xs)[0])] ## for non 1d features 
#print ("train_feature_shape:",np.shape(X))
	




# X=feature_Xs 


data_Y,label_Y=generate_data(valid_imagePaths_list,valid_label,valid_num,valid_batch_size,resize_w,resize_h,max_label) 
feature_Y=Encoder.predict(data_Y, batch_size=None, verbose=0, steps=None)


# feature_Y=feature_Y[:,:,:,5:15]
print ("train_feature_shape:",np.shape(feature_Y))

Y=[feature_Y[i,:,:,:].flatten('C') for i in np.arange(np.shape(feature_Y)[0])]
print ("train_feature_shape:",np.shape(Y))
print ("label_Y_shape:",np.shape(label_Y))



###### KNN train and prediction ############
KNN = KNeighborsClassifier(n_neighbors=10)
KNN.fit(X, label_Xs) 


prediction=KNN.predict(Y) ### the input should be a [13*13*16] 1d-vector 
print ('pred_shape:',np.shape(prediction))
print('label_Y')
print (label_Y)
print('prediction')
print (prediction)


top1_accuracy=np.true_divide(np.sum(label_Y==prediction),len(label_Y)) 
print ('top 1 accuracy:',top1_accuracy)

class_probs=KNN.predict_proba(Y)
top5_right=[]
i=0 
for class_prob in class_probs: 
	top5_prob_label=sorted(range(len(class_prob)), key=lambda i: class_prob[i])[-5:] 
	#print ('top 5,pred',top5_prob_label,prediction[i])
	print ('top 5 prob',sorted(class_prob)[-5:])
	if label_Y[i] in top5_prob_label: 
		top5_right.append(1)
	else:
		top5_right.append(0)
	i+=1
top5_accuracy=np.true_divide(np.sum(top5_right),len(label_Y)) 
print ('top 5 accuracy:',top5_accuracy)





### for feature check####

# for i in (14,15):
# 	print (valid_imagePaths_list[i])
# 	img2check=i ### the image in test set to check 
# 	# img_orig= cv2.imread(valid_imagePaths_list[img2check])[:,:,1]
# 	# img_orig= cv2.resize(img_orig,(128,128))/255  ### -2 for maxpool and upsample commendation 
# 	feature_pred_test=feature_pred[img2check,:,:,5]
# 	feature_pred_test=np.squeeze(feature_pred_test)
# 	img_pred_test=np.divide(feature_pred_test-np.min(feature_pred_test),np.max(feature_pred_test)-np.min(feature_pred_test))
# 	# print (feature_pred_test-np.min(feature_pred_test))
# 	print (img_pred_test)
# 	#cv2.imshow('orig',img_orig)
# 	cv2.imshow('recons',img_pred_test)
# 	cv2.waitKey()





