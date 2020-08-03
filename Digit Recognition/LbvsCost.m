function[yes] = LbvsCost(InitTheta, X, y)
iter=20;
const=.005;
lambda=zeros(iter, 1);
lambda(1,1)=.1;

for c=2:iter
    lambda(c,1)=lambda(c-1, 1)+const;
end

Jvector=zeros(iter, 1);

for i = 1:iter
    options=optimset('MaxIter', 10);
    CostFunction = @(p) Prop(p, X, y, 400,25,10, lambda(i,1));
    [Theta, ~]=fmincg(CostFunction, InitTheta, options);
    
    [J, ~]=Prop(Theta, X, y, 400, 25, 10, lambda(i, 1));
    Jvector(i,1)=J;
end
% fprintf("\n %f\n", Jvector);
% fprintf("\n %f\n", lambda);
plot(lambda, Jvector, 'r-.');
end

