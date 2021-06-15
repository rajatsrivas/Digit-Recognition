function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));%25*401
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));%10*26
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m,1), X];%5000*401

Z1 = Theta1*X';
a2 = sigmoid(Z1);%25*5000
a2 = [ones(m,1), a2'];%5000*26

Z2 = Theta2*a2'; %10*26 * 26*5000 10*5000
h = (sigmoid(Z2)); %10*5000

y_new = zeros(num_labels,m); %5000*10
for i =1:m,
	y_new(y(i),i)=1;
	end
	
J = (1/m)*sum(sum((-y_new).*log(h)-((1-y_new).*log(1-h))))+(lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

for t=1:m
	a_1 = (X(t,:))';%401*1
	
	z_2 = Theta1 * a_1; %25*1
	a_2 = sigmoid(z_2); %25*1
	a_2 = [1;a_2]; %26*1
	
	z_3 = Theta2 * a_2; %10*1
	a_3 = sigmoid(z_3); %10*1
	
	delta_3 = a_3 - y_new(:,t); %10*1
	
	z_2 = [1;z_2];
	delta_2 = (Theta2' * delta_3).*sigmoidGradient(z_2); %26*1
	delta_2 = delta_2(2:end); %25*1
	
	Theta1_grad = Theta1_grad + delta_2*a_1';
	Theta2_grad = Theta2_grad + delta_3*a_2';
	end
	
Theta1_grad= (1/m)*Theta1_grad;
Theta2_grad= (1/m)*Theta2_grad;

Theta1_grad(:,2:end)= Theta1_grad(:,2:end) + ((lambda/m)*Theta1(:,2:end));
Theta2_grad(:,2:end)= Theta2_grad(:,2:end) + ((lambda/m)*Theta2(:,2:end));

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
