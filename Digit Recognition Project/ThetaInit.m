function W = ThetaInit(L_Out, L_In)

epsilon = 0.12;
W = rand(L_Out, 1 + L_In) * 2 * epsilon - epsilon;



end

