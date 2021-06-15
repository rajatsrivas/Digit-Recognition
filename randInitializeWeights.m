function W = randInitializeWeights(L_in, L_out)
W = zeros(L_out, 1 + L_in);
e= 0.12;
W = rand(size(W)) * 2 * e - e;
end