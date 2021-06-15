function p = predict(Theta1, Theta2, X)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];
t1 = sigmoid(X * Theta1');
t1 = [ones(m, 1) t1];

t2 = sigmoid( t1 * Theta2');

[~, p] = max(t2, [], 2);

end