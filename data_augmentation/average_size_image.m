path=pwd;
path_reference=strcat(path,'\..\..\FID-300\references')
path =strcat(path,'\..\..\FID-300\tracks_cropped');
N = 300; % set number of croped image
offset=10; % set minimum offset of cropped image
S_r = dir(fullfile(path_reference,'*.png'));
S=dir(fullfile(path,'*.jpg'));
load ..\..\FID-300\label_table;
fileID_test = fopen('label_test.txt', 'w'); % file to save the label for cropped image
fileID_train = fopen('label_train.txt', 'w'); % file to save the label for cropped image
height=0;
width=0;
%compute the average size of the reference images
for l =1:numel(S_r)
    F = fullfile(path_reference,S_r(l).name);
    I = imread(F); %read image
    [H,W,~]= size(I);
    height=height+H;
    width=width+W;
end 
PH=round(height/numel(S_r));
PW=round(width/numel(S_r));
for l =1:numel(S) % total number of images in the folder
    
    F = fullfile(path,S(l).name);
    I = imread(F); %read image
    %J = imresize(I,[PH PW])
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

    % preprocess the Image
    test_data_index=unique(randi([1,100],[1,20]));
    for j=1:100
        %set the name for cropped images
        index=strcat(strcat(int2str(l),'_'),int2str(j));
        if ismember(j,test_data_index)
            patch_name=strcat(strcat(path, '/cropped/test/'), index);
            fileID=fileID_test;
        else 
            patch_name=strcat(strcat(path, '/cropped/train/'), index);
            fileID=fileID_train;
        end
        %generate number of modifications for each cropped image
        no_image_change=randi([0 4]);
        patch=imrotate(I,3.6*j);
        for i=1:no_image_change
            % generate the type of modification 
            % 1: affine
            % 2: flip 
            % 3: brightness
            % 4: guassian
            method_image_change=randi([1,4]);
            switch method_image_change
                case 1
                    %affine transform
                    tform = affine2d([1 randi([0,50])/100 0;randi([0,50])/100 1 0; 0 0 1]);
                    patch = imwarp(patch,tform);               
                case 2
                    % find the flip direction 
                    % 1: horizontal flip 
                    % 2: vertical flip
                    flip_type=randi([1,2]);
                    patch=flip(patch, flip_type);
                case 3
                    % find the change of brightness between [-45, 45]
                    brightness=randi([-25,25]);
                    patch=patch+brightness;
                case 4
                    % B = imgaussfilt(A,SIGMA) filters image A with a 2-D Gaussian smoothing
                    % kernel with standard deviation specified by SIGMA. SIGMA can be a
                    % scalar or a 2-element vector with positive values. If sigma is a
                    % scalar, a square Gaussian kernel is used
                    patch=imgaussfilt(patch,2);
            end
        end       
        J = imresize(I,[PH PW]);
        fprintf(fileID,'%s, %i\n',index, label_table(l,2)); % write the label of cropped image
        imwrite(patch,strcat(patch_name,'.jpg'));
    end
end

fclose(fileID);