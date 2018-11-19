#### data pre-process 



#### Detection procedure ####
1. train Feature extractor: 
	Input images(train+reference) -> AE -> output images(input images) 
                                         |
                                     get features

2. train Classifier:
		input image -> trained feature extractor -> output features ->train: 1. CNN classifier -> image labels
									          OR 2. kNN classifier 	



### current progress ### 
Extractor (Tong)
1. input random (need to load large image files)
2. auto-encoder (baisc structure tested) : load batched data, save model, load model,prediction (tested)
3. get encoder output(feature extractor tested): extract encoder part of AE, encoder prediction(feature extraction)(tested) 

Classifier: 
Classifier (haven't started)  


### current problems ###
1. Load large number of images: could solve by: shuffle img paths and load imgs accordingly, need to let keras keep training (Tong)
	   