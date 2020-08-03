function h = sigmoidGrad(z)
h=zeros(size(z));
h=sigmoid(z).*(1-sigmoid(z));
end

