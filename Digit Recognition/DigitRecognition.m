%DIGIT RECOGNITION PROJECT
clear ; close all; clc
%Obtaining Data and Define Variables
load('ex4data1.mat');
load('imgdata.mat');

RX=imgdata;

rnd=randperm(length(X));
X=X(rnd, :);
y=y(rnd, :);

Xtrain=X(1:3000, :);
ytrain=y(1:3000, 1);

Xval=X(3001:4000, :);
yval=y(3001:4000, :);

Xtest=X(4001:end, :);
ytest=y(4001:end, :);

inputsize=400;
hiddenlayersize=25;
resultamount=10; 

%for fmincg
Init1= ThetaInit(hiddenlayersize, inputsize); 
Init2= ThetaInit(resultamount, hiddenlayersize);
InitTheta= [Init1(:); Init2(:)]; 

lambda=.11;
options=optimset('MaxIter', 50);
CostFunction = @(p) Prop(p, Xtrain, ytrain, inputsize, hiddenlayersize, resultamount, lambda);

[nn_params, cost]=fmincg(CostFunction, InitTheta, options);

Theta1=reshape(nn_params(1:hiddenlayersize*(inputsize+1)), hiddenlayersize, (inputsize+1));
Theta2=reshape(nn_params(1+hiddenlayersize*(inputsize+1):end), resultamount, (hiddenlayersize+1));

pred=predict(Theta1, Theta2, Xtest);
fprintf('\nSet Accuracy: %f\n', mean(double(pred == ytest)) * 100);

