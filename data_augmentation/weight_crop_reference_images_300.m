path=pwd;
path_reference=strcat(path,'/../../FID-300/references')
path =strcat(path,'/../../FID-300/tracks_cropped');
%path_reference=strcat(path,'/references')
%path =strcat(path,'/tracks_cropped');
N = 300; % set number of croped image
offset=10; % set minimum offset of cropped image
S_r = dir(fullfile(path_reference,'*.png'));
S=dir(fullfile(path,'*.jpg'));
load ../../FID-300/label_table;
%load label_table;
fileID_test = fopen('label_test.txt', 'w'); % file to save the label for cropped image
fileID_test_index = fopen('label_test_index.txt', 'w'); % file to save the index for cropped image
fileID_train = fopen('label_train.txt', 'w'); % file to save the label for cropped image
fileID_train_index = fopen('label_train_index.txt', 'w'); % file to save the index for cropped image
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
%Generate training data from the reference images
for l =1:numel(S_r)
    if ismember(l,label_table(:,2))
        F = fullfile(path_reference,S_r(l).name);
        I = imread(F); %read image

        for j=1:50
            %set the name for cropped images
            index=strcat(strcat(int2str(l),'_'),int2str(j));
            index=strcat('r',index);
            patch_name=strcat(strcat(path, '/cropped/train/'), index);

            %generate number of modifications for each cropped image
            no_image_change=randi([1 5]);
            method_image_change=randperm(5,no_image_change);
            patch=I;
            for i=1:no_image_change
                % generate the type of modification 
                % 1: scale
                % 2: flip 
                % 3: brightness
                % 4: guassian
                % 5: rotation
                switch method_image_change(i)
                    case 1
                        %scale 
    %                     [H,W,~]= size(patch);
    %                     H1=H*randi([7,15])/10;
    %                     W1=W*randi([7,15])/10;
    %                     patch = imresize(patch,[H1 W1]);
    %                     a=floor(max(1,H1-H));
    %                     b=floor(max(1,W1-W));
                        patch = patch;%imcrop(patch,[randi([1 b],1,1),randi([1 a],1,1),W,H]);            

                    case 2
                        % find the flip direction 
                        % 1: horizontal flip 
                        % 2: vertical flip
                        flip_type=randi([1,2]);
                        patch=flip(patch, flip_type);
                    case 3
                        % find the change of brightness between [-45, 45]
                        brightness=randi([-45,0]);
                        patch=patch+brightness;
                    case 4
                        % B = imgaussfilt(A,SIGMA) filters image A with a 2-D Gaussian smoothing
                        % kernel with standard deviation specified by SIGMA. SIGMA can be a
                        % scalar or a 2-element vector with positive values. If sigma is a
                        % scalar, a square Gaussian kernel is used
                        sigma=randi([12,18])/500.0;
                        patch=imgaussfilt(patch,sigma);
                    case 5
                        %rotation
                        angle_type=randi([0 1]);
                        [h,w]=size(patch);
                        width=max(h,w);
                        patch=imresize(patch,[width,width]);
                        theta=0;
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

            J = imresize(patch,[PH PW]);
            fprintf(fileID_train,'%i,', l); % write the label of cropped image
            fprintf(fileID_train_index,' %s,', index); % write the index of cropped image
            imwrite(J,strcat(patch_name,'.jpg'));
        end
    end
end 
test_data_index=randperm(300,60);
for l =1:numel(S) % total number of images in the folder
    
    F = fullfile(path,S(l).name);
    I = imread(F); %read image
    
    %generate test images
    if ismember(l,test_data_index)
        J = imresize(I,[PH PW]);
        patch_name=strcat(strcat(path, '/cropped/test/'), int2str(l));
        fprintf(fileID_test,' %i,', label_table(l,2)); % write the label of cropped image
        fprintf(fileID_test_index,' %i,', l); % write the index of cropped image
        imwrite(J,strcat(patch_name,'.jpg'));
    else

        for j=1:50
            %set the name for cropped images
            index=strcat(strcat(int2str(l),'_'),int2str(j));

            patch_name=strcat(strcat(path, '/cropped/train/'), index);

            %generate number of modifications for each cropped image
            
            no_image_change=randi([1 5]);
            method_image_change=randperm(5,no_image_change);
            patch=I;
            for i=1:no_image_change
                % generate the type of modification 
                % 1: scale
                % 2: flip 
                % 3: brightness
                % 4: guassian
                % 5: rotation
                switch method_image_change(i)
                    case 1
                        %scale
%                         [H,W,~]= size(patch);
%                         H1=H*randi([7,15])/10;
%                         W1=W*randi([7,15])/10;
%                         patch = imresize(patch,[H1 W1]);
%                         a=floor(max(1,H1-H));
%                         b=floor(max(1,W1-W));
%                         patch = imcrop(patch,[randi([1 b],1,1),randi([1 a],1,1),W,H]);               
                          patch =patch;

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
                 
                        sigma=randi([12,18])/500.0;
                        patch=imgaussfilt(patch,sigma);
                    case 5
                        %rotation
                        angle_type=randi([0 1]);
                        [h,w]=size(patch);
                        width=max(h,w);
                        patch=imresize(patch,[width,width]);
                        theta=0;
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
            
            J = imresize(patch,[PH PW]);
            %fprintf(fileID_train,'%s, %i\n',index, label_table(l,2)); % write the label of cropped image
            fprintf(fileID_train,'%i,', label_table(l,2)); % write the label of cropped image
            fprintf(fileID_train_index,' %s,', index); % write the index of cropped image
            imwrite(J,strcat(patch_name,'.jpg'));
        end
    end
end

fclose(fileID_train);
fclose(fileID_test);