function [w, b, obj, cvErrs] = naiveLOOCV_ridgeReg(X1, y, lamda)
k = size(X1,1);
n = size(X1,2);
X = [X1;ones(1,n)];
I_k = eye(k);
O_k = zeros(k,1);
I = [I_k, O_k; O_k', 0];
w = pinv(X*X'+lamda.*I)*X*y;
b = w(k+1);
obj = lamda*(w')*w + (w'*X - y')*(w'*X - y')';
cvErrs = zeros(n,1);
for i=1:n
  xi = X(:,i);
  Xi = [X(:,1:i-1), X(:,i+1:n)];
  yi = [y(1:i-1);y(i+1:n)];
  wi = pinv(Xi*Xi'+lamda.*I)*Xi*yi;
  cvErrs(i) = wi'*xi-y(i);
end
end