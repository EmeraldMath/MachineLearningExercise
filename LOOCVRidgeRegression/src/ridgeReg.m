function [w, b, obj, cvErrs] = ridgeReg(X1, y, lamda)
k = size(X1,1);
n = size(X1,2);
X = [X1;ones(1,n)];
I_k = eye(k);
O_k = zeros(k,1);
I = [I_k, O_k; O_k', 0];
C = X*X' + lamda.*I;
invC = pinv(C);
w = invC*X*y;
b = w(k+1);
obj = lamda*(w')*w + (w'*X - y')*(w'*X - y')';
cvErrs = zeros(n,1);
for i=1:n
  xi = X(:,i);
  cvErrs(i) = (w'*xi - y(i))/(1 - xi'*invC*xi);
end
end

  
