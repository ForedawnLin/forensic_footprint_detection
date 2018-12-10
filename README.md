#### data augmentation

#### data pre-process 



#### Detection procedure ####
1. train Feature extractor: 
	Input images(train+reference) -> AE -> output images(input images) 
                                         |
                                     get features

2. train Classifier:
		input image -> trained feature extractor -> output features ->train: 1. CNN classifier -> image labels
									          OR 2. kNN classifier 	

#### PCA ####
Applying eigenface to extract 100 eigen footprints from the training set. Then use 100 eigen footprints for testing.

### current progress ### 
Extractor (Tong)
1. input random (need to load large image files) (solved) 
2. auto-encoder (baisc structure tested) : load batched data, save model, load model,prediction (tested)
3. get encoder output(feature extractor tested): extract encoder part of AE, encoder prediction(feature extraction)(tested) 

Classifier: 
Classifier (Added AE_CNN classifier) (Tong)
1. Classifier 1: train AE_CNN_classifier: two outputs: 1. MSE loss for AE 2. crossentropy(CE) loss for classification   

PCA (Tianlong)
PCA comparison added by adapting eigenface. (Tianlong)

### current problems ###
1. (solved)Load large number of images: could solve by: shuffle img paths and load imgs accordingly, need to let keras keep training (Tong)
	   
