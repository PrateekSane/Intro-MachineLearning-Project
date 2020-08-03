function [J, grad] = Prop(params, X, y, inputsize, hiddenlayersize, resultamount, lambda)

eye_matrix = eye(10);
y_matrix = eye_matrix(y,:);

Theta1=reshape(params(1:hiddenlayersize*(inputsize+1)), hiddenlayersize, (inputsize+1));
Theta2=reshape(params(1+hiddenlayersize*(inputsize+1):end), resultamount, (hiddenlayersize+1));

m=size(X,1);

a1= [ones(m, 1) X];
z2=a1*Theta1';
a2= sigmoid(z2);
a2=[ones(m,1) a2];
z3= a2*Theta2';
a3=sigmoid(z3);


J=(sum(sum((-y_matrix).*(log(a3))-(1-y_matrix).*log(1-a3))))/m;

J=J+(sum(sum(Theta2(:,2:end).^2)))*(lambda/(2*m));

del3= a3-y_matrix;
del2= (del3*Theta2(:, 2:end)).*sigmoidGrad(z2);
grad1=(del2'*a1)./m;
grad2=(del3'*a2)./m;

grad=[grad1(:); grad2(:)];

end


