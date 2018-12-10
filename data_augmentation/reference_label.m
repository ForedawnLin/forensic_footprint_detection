path=pwd;
path_reference=strcat(path,'/../../FID-300/references')
path =strcat(path,'/../../FID-300/tracks_cropped');

S_r = dir(fullfile(path_reference,'*.png'));

load ../../FID-300/label_table;

fileID_reference = fopen('label_reference.txt', 'w'); % file to save the label for cropped image
fileID_reference_index = fopen('label_reference_index.txt', 'w'); % file to save the index for cropped image

%compute the average size of the reference images
height=0;
width=0;
for l =1:numel(S_r)
    F = fullfile(path_reference,S_r(l).name);
    I = imread(F); %read image
    [H,W,~]= size(I);
    height=height+H;
    width=width+W;
end 
h_ave=round(height/numel(S_r));
w_ave=round(width/numel(S_r));


for l =1:numel(S_r) % total number of images in the folder    
    F = fullfile(path_reference,S_r(l).name);
    I = imread(F); %read image
    J = imresize(I,[h_ave w_ave]);
    reference_name=strcat(strcat(path, '/cropped/reference/'), int2str(l));
    fprintf(fileID_reference,' %i,', l); % write the label of cropped image
    fprintf(fileID_reference_index,' %i,', l); % write the index of cropped image
    imwrite(J,strcat(reference_name,'.jpg'));
end

fclose(fileID_reference);
fclose(fileID_reference_index);
fclose('all');

