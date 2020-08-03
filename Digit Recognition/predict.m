function p = predict(Theta1, Theta2, X, y)
m=size(X,1);

h1=sigmoid([ones(m, 1) X]*Theta1');
h2= sigmoid([ones(m,1) h1]*Theta2');

[bla, p]=max(h2, [], 2);
if(p==10)
    p=0;

    
%fprintf('\nSet Accuracy: %f\n', mean(double(p == y)) * 100);

   
end

