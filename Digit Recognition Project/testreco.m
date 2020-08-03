
clear ; close all; clc
inputsize=400;
hiddenlayersize=25;
resultamount=10;      % (note that we have mapped "0" to label 10)
% Load the weights into variables Theta1 and Theta2
load('ex4data1.mat');

Xtrain=X(1:3000, :);
ytrain=y(1:3000, 1);

Xval=X(3001:4000, :);
yval=y(3001:4000, :);

Xtest=X(4001:end, :);
ytest=y(4001:end, :);


% Unroll parameters 
%nn_params = [Theta1(:) ; Theta2(:)];
Init1= ThetaInit(hiddenlayersize, inputsize); 
Init2= ThetaInit(resultamount, hiddenlayersize);
InitTheta= [Init1(:); Init2(:)]; 
options = optimset('MaxIter', 50);

costFunction = @(p) Prop(p, Xtrain, ytrain, inputsize, hiddenlayersize,resultamount);
[nn_params, cost] = fminunc(costFunction, InitTheta, options);
fprintf('%f\n', Cost);

