function [H, f, A, b, Aeq, beq, lb, ub] = setPar(X, Y, C)
n = size(X,2);
H = zeros(n,n);
for i = 1:n
    for j = 1:n
        H(i,j) = Y(i)*kernel(X(:,i),X(:,j))*Y(j);
    end
end
f = repmat(-1,[n,1]);
A = [];
b = [];
Aeq = Y';
beq = 0;
lb = zeros(n,1);
ub = repmat(C,[n,1]);
end

function y = kernel(xi,xj)
y = xi'*xj;
end




