%function hw4_2_6()
 %[0.685185185185185,0.660493827160494,0.666666666666667,0.666666666666667]
 %C = [0.01, 0.1,1, 10];
%  step = size(C,2);
%  T_c = zeros(1,step);
%  A_c = zeros(1,step);
%  for p = 1:step
%  X = [trD(:,800:1100), trD(:,1811:2360)];
%  Y = [trLb(800:1100,:);trLb(1811:2360,:)];
%  vX = [valD(:,220:330), valD(:, 600:650)];
%  vY = [valLb(220:330,:); valLb(600:650,:)];
%  iter = 3;
%  gamma = kPar(X, X);
%  [alpha, b, gamma] = trainModel(X, Y, iter, C(p));
%  val_pred = classifier(alpha, b, X, Y, vX, iter, gamma);
%  val_acc = accuracy(val_pred, vY);
%  fprintf('Validation Accuracy: %f\n', val_acc);
%  train_pred = classifier(alpha, b, X, Y, X, iter, gamma);
%  train_acc = accuracy(train_pred, Y);
%  fprintf('Train Accuracy: %f\n', train_acc);
%  T_c(p) = 1 - train_acc;
%  A_c(p) = 1 - val_acc;
%  end
 clear;
 load('q2_2_data.mat');
 iter = 10;
 C = 0.1;
 gamma = kPar(trD, trD);
 [alpha, b, gamma] = trainModel(trD, trLb, iter, C);
 val_pred = classifier(alpha, b, trD, trLb, valD, iter, gamma);
 val_acc = accuracy(val_pred, valLb);
 fprintf('Validation Accuracy: %f\n', val_acc);
 test_pred = classifier(alpha, b, trD, trLb, tstD, iter, gamma);
 save('C01par.mat','alpha','b');
 save('C01predict.mat','test_pred');
 save('valAcc.mat','val_acc');
 
% end

function acc = accuracy(pred, act)
n = size(pred,1);
consist = act(pred == act);
acc = size(consist,1)*1.0/n; 
end

function pred = classifier(alpha, b, trD, trLb, valD, iter, gamma)
n = size(trD,2);
vn = size(valD,2);
pred = zeros(vn,1);
for i = 1:vn
    dist = zeros(iter, 1);
    for it = 1:iter
        Y = repmat(-1,[n,1]);
        Y(trLb == it) = 1;
        Y = double(Y);
        sum_it = 0;
        for j = 1:n
            sum_it = sum_it + alpha(j,it)*Y(j)*kernel(trD(:,j), valD(:,i), gamma);
        end
        dist(it) = sum_it+b(it);
    end
    [~, id] = max(dist);
    pred(i) = id;
end
end

function [alpha, b, gamma] = trainModel(X, trLb, iter, C)
n = size(X,2);
alpha = zeros(n,10);
b = zeros(n,1);
for it = 1:iter
Y = repmat(-1,[n,1]);
Y(trLb == it) = 1;
Y = double(Y);
[H, f, A, b2, Aeq, beq, lb, ub, gamma] = setPar(X, Y, C);
[alpha(:,it), ~] = quadprog(H,f,A,b2,Aeq,beq,lb,ub);
col = alpha(:,it);
val = col(col > 1e-9 & col < C - 1e-9);
k = find(alpha(:,it) == val(1), 1);
tmp = 0;
    for j = 1:n
        tmp = tmp + alpha(j,it)*Y(j)*kernel(X(:,j),X(:,k),gamma);
    end
b(it) = Y(k) - tmp;
end
end


function gamma = kPar(X1, X2)
n = size(X1,2);
m = size(X2,2);
p = zeros(1,n*n);
t = 1;
for i = 1:n
    for j = 1:m
        p(t) = (X1(:,i) - X2(:,j))'*(X1(:,i) - X2(:,j));
        t = t + 1;
    end
end
gamma = mean(p);
end

function [H, f, A, b, Aeq, beq, lb, ub, gamma] = setPar(X, Y, C)
n = size(X,2);
H = zeros(n,n);
gamma = kPar(X, X);
for i = 1:n
    for j = 1:n
        H(i,j) = Y(i)*kernel(X(:,i),X(:,j),gamma)*Y(j);
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

function y = kernel(xi,xj,gamma)
%y = xi'*xj;
y = exp(-(xi-xj)'*(xi-xj)*1.0/gamma);
end