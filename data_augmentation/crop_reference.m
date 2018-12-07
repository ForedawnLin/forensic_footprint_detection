PH = 240; % set size of croped image
PW = 100; % set size of croped image
path=pwd;
path_reference=strcat(path,'/../../FID-300/references')
%path =strcat(path,'/references');
N = 50; % set number of croped image

S = dir(fullfile(path,'*.png'));
load ../../FID-300/label_table;
%load label_table;
fileID_train = fopen('label_train.txt', 'w'); % file to save the label for cropped image
fileID_train_index = fopen('label_train_index.txt', 'w'); % file to save the index for cropped image
%compute the average size of the reference images
height=0;
width=0;
for l =1:numel(S)
    F = fullfile(path,S(l).name);
    I = imread(F); %read image
    [H,W,~]= size(I);
    height=height+H;
    width=width+W;
end 
h_ave=round(height/numel(S));
w_ave=round(width/numel(S));
for l =1:numel(S) % total number of images in the folder
    
    F = fullfile(path,S(l).name);
    I = imread(F); %read image
    [H,W,~]= size(I);

    Y = randperm(H-PH,N); % generate random crop coordinates
    X = randperm(W-PW,N);
    coordinates=unique([X; Y]','rows');
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 

    % Cropping the Image
    
    for j = 1:N
        patch = imcrop(I, [coordinates(j,1), coordinates(j,2),randi([PW,W-coordinates(j,1)]),randi([PH,H-coordinates(j,2)])]);
        index=strcat(strcat(int2str(l),'_'),int2str(j));
       
        patch_name_no=strcat(strcat(path, '/cropped/train_noise/'), index);
        patch_name=strcat(strcat(path, '/cropped/train/'), index);
        %generate number of modifications for each cropped image
        no_image_change=randi([0 3]);
        method_image_change=randperm(3,no_image_change);
        
        for i=1:no_image_change
            % generate the type of modification          
            % 1: flip 
            % 2: brightness
            % 3: rotation
            switch method_image_change(i)
                case 1
                    % find the flip direction 
                    % 1: horizontal flip 
                    % 2: vertical flip
                    flip_type=randi([1,2]);
                    patch=flip(patch, flip_type);
                case 2
                    % find the change of brightness between [-45, 45]
                    brightness=randi([-45,0]);
                    patch=patch+brightness;
                case 3 
                    %rotation
                    angle_type=randi([0 1]);
                    [h,w]=size(patch);
                    width=max(h,w);
                    patch=imresize(patch,[width,width]);
                    if angle_type==0
                        theta=randi([-45,45]);                            
                        patch=imrotate(patch,theta);
                        theta=abs(theta);
                    else
                        theta=randi([135,225]);
                        patch=imrotate(patch,theta);
                        theta=abs(theta-180);
                    end
                    angle=theta/180*pi+pi/4;
                    if angle>pi
                        angle=angle-pi/2;
                    end
                    new_width=width/(sqrt(2)*sin(angle)); % width of cropped area
                    [h,w]=size(patch); %size of rotated image
                    patch = imcrop(patch,[h/2-new_width/2,w/2-new_width/2,new_width,new_width]);
            end
        end       

        J = imresize(patch,[h_ave w_ave]);
        fprintf(fileID_train,'%i,', l); % write the label of cropped image
        fprintf(fileID_train_index,' %s,', index); % write the index of cropped image
        imwrite(J,strcat(patch_name,'.jpg'));
        
        % add noise
        no_image_change=randi([1 4]);
        method_image_change=randperm(4,no_image_change);
        for i=1:no_image_change
            switch method_image_change(i)
                case 1 
                    J = imnoise(J,'gaussian');
                case 2
                    J = imnoise(J,'poisson');
                case 3
                    J = imnoise(J,'salt & pepper');
                case 4
                    J = imnoise(J,'speckle');                 
            end
        end
        imwrite(J,strcat(patch_name_no,'.jpg'));
    end

end

fclose(fileID_train);
fclose(fileID_train_index);
