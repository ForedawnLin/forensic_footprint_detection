from __future__ import print_function

from time import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn import preprocessing

# from keras.layers import Input

#import skimage
#from skimage import io

# print(__doc__)

### load data ###
# mainPath='../../FID-300/tracks_cropped/cropped/train/'  ## main path of the pictures 
# TODO: change the path to augumented figures
mainPath='../../FID-300/tracks_cropped/cropped/train/'  ## main path of the pictures 
pic_fmt='.jpg'  ## picture format 
imagePaths_list=[]  ## init image paths
train_index_path='../data_augmentation/label_train_index.txt' 
train_label_path= '../data_augmentation/label_train.txt'
FileID_train_index=open(train_index_path,'r')
for indice in FileID_train_index:
	indice=indice.replace(' ','')
	indice=indice.replace('r','') # remove r which is inconsistent with image names
	index=indice.split(',')
index=index[:-1]
# print (index)
FileID_train_label=open(train_label_path,'r')
for labels in FileID_train_label:
	label=labels.split(',')
label=label[:-1]
#print (label)
imagePaths_list=[mainPath+index[i]+pic_fmt for i in np.arange(len(index))]
# print ('imagePaths_list',imagePaths_list)
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
# img = cv2.imread(imagePaths_list[0])[:,:,1]
# print('path', imagePaths_list[0])
# img = cv2.imread(imagePaths_list[1], 0)
img = cv2.imread(train_imagePaths_list[0], 0)
# cv2.imshow('image',img)
img_h = np.shape(img)[0] ### input layer image height 
img_w = np.shape(img)[1] ### input layer image width
img_channel=1 ### input layer image width, gray image 	
print ('imag_shape',train_imagePaths_list[30000], img_h,img_w,img_channel)

img_reshape = img.reshape( (img_h*img_w, 1) )
print ('imag_reshape',np.shape(img_reshape)[0],np.shape(img_reshape)[1])

#resize_w=128; ### resize image to before feeding into network 
#resize_h=128;
#input_img = Input(shape = (resize_w, resize_h, img_channel)) ### -2 for maxpool and upsample commendation 

# TODO fix data error: training img is less than 30000
train_num = 500
#imgs = []
#imgs = np.array(imgs)
imgs = np.zeros( (train_num, img_h*img_w) )  
for i in range(0, train_num):
	img = cv2.imread(train_imagePaths_list[i], 0)
	img_h = np.shape(img)[0] ### input layer image height 
	img_w = np.shape(img)[1] ### input layer image width
	img_channel=1 ### input layer image width, gray image 	
	print ('imag_shape',i, img_h,img_w,img_channel)

	img_reshape = img.reshape( (img_h*img_w, 1) )[0]
	# print ('imag_reshape',np.shape(img_reshape)[0],np.shape(img_reshape)[1])
	# imgs.append(img_reshape)
	# imgs = np.append(imgs, img_reshape, axis = 0)
	imgs[i] = img_reshape
	print ('imgs_shape after', imgs.shape)

# Display progress logs on stdout
# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# #############################################################################
# Download the data, if not already on disk and load it as numpy arrays
### lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
# n_samples, h, w = lfw_people.images.shape
# imgs = np.array(imgs)

n_samples, n_pix = imgs.shape
h = img_h
w = img_w
print ('imgs_shape',n_samples, n_pix, h, w)

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
### X = lfw_people.data
X = imgs
# normalize the data to [0 1]
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

n_features = X.shape[1]

# the label to predict is the id of the person
# y = lfw_people.target
y = train_label[0:train_num]


# target_names = lfw_people.target_names
# n_classes = target_names.shape[0]
# target_names = train_label[0:12]
# n_classes = 1175 # number of reference images is the number of classes
n_classes = 5 # number of reference images is the number of classes

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

# #############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# #############################################################################
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

#print(classification_report(y_test, y_pred, target_names=target_names))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

# def title(y_pred, y_test, target_names, i):
def title(y_pred, y_test, i):
    #pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    #true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    pred_name = y_pred[i]
    true_name = y_test[i]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

# prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
prediction_titles = [title(y_pred, y_test, i) for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()