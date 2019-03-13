
function [w, b, obj, Errs] = train_ridgeReg(X1, y, lamda)
k = size(X1,1);
n = size(X1,2);
X = [X1;ones(1,n)];
%w = zeros(size(X,1));
%normal equation
I_k = eye(k);
O_k = zeros(k,1);
I = [I_k, O_k; O_k', 0];
w = pinv(X*X'+lamda.*I)*X*y;
b = w(k+1);
obj = lamda*(w(1:k)')*w(1:k) + (w'*X - y')*(w'*X - y')';
%Errs = zeros(1,n);
Errs = X'*w-y;
end