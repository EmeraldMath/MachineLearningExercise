function [w, b] = primal(alpha, X, Y, C)
[d,n] = size(X);
w = zeros(d,1);
for i = 1:n
    w = w + alpha(i)*Y(i)*X(:,i);
end
val = alpha(alpha > 1e-9 & alpha < C - 1e-9);
k = find(alpha == val(1),1);
b = Y(k) - w'*X(:,k);
end
