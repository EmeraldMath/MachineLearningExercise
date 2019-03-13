run ./vlfeat-0.9.21/toolbox/vl_setup;
[w,b] = train_model();
HW4_Utils.genRsltFile(w, b, 'val', './hw4_3_4/hw4_3_4_1');
[ap, prec, rec] = HW4_Utils.cmpAP('./hw4_3_4/hw4_3_4_1', 'val');
fprintf('AP %f',ap);
saveas(gcf,'./hw4_3_4/APinIteration.png');
function [w,b] = train_model()
clear;
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
%load('trD.mat');
%C = 0.1;
C = 10;
X = trD;
Y = trLb;
[H, f, A, b2, Aeq, beq, lb, ub] = setPar(X, Y, C);
[alpha, dual_min] = quadprog(H,f,A,b2,Aeq,beq,lb,ub);
[w, b] = primal(alpha, X, Y, C);

[d, vn] = size(valD);
vio = 0;
obj = 0.5*w'*w;
fp = 0;
fn = 0;
tp = 0;
tn = 0;
for i = 1:vn
    sign = w'*valD(:,i)+b;
    zeta = 1 - valLb(i)*sign;
    if (zeta > 1e-15)
        vio = vio + 1;
        obj = obj + C*zeta;
    end
    if (sign > 1e-15)
        if (valLb(i) == 1)
            tp = tp + 1;
        else
            fp = fp + 1;
        end
    end
    if (sign < -1e-15)
        if (valLb(i) == -1)
            tn = tn + 1;
        else
            fn = fn + 1;
        end
    end
end

% acc = (vn-fp-fn)*1.0/vn;
% fprintf('C = %f\n', C);
% fprintf('Accuracy: %f\n',acc);
% fprintf('Objective value of SVM: %f\n', obj);
% fprintf('Number of support vectors: %d\n', vio);
% fprintf('Confusion Matrix:\n');
% fprintf('\t predicted +\tpredicted -\tsum\n');
% fprintf('actual +%8d\t%7d \t%3d\n',tp, fn, sum(valLb==1));
% fprintf('actual -%8d\t%7d \t%3d\n',fp, tn, sum(valLb==-1));
% fprintf('sum     %8d\t%7d\n',tp+fp, fn+tn);
end

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