function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve

% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i =1:m
 
  [theta] = trainLinearReg(X(1:i,:), y(1:i), lambda);

  [J1, grad1] = linearRegCostFunction(X(1:i,:), y(1:i), theta,0);

  [J2, grad2] = linearRegCostFunction(Xval, yval, theta,0);

  error_train(i,1)=J1;
  error_val(i,1)=J2;

 endfor

end
