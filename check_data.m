clc 
clear all

filePath='../FID-300/label_table.mat';
data=load(filePath);
data=data.label_table;
counts=zeros(1175,1); 
for i=1:1175  %%% 1175 is the total label number 
    counts(i)=sum(data(:,2)==i);
end 
plot(1:length(counts),counts,'ro'); 