clear;
load('q2_1_data.mat');
%C = 0.1;
C = 10;
X = trD;
Y = trLb;
[H, f, A, b2, Aeq, beq, lb, ub] = setPar(X, Y, C);
[alpha, dual_min] = quadprog(H,f,A,b2,Aeq,beq,lb,ub);
[w, b] = primal(alpha, X, Y, C);

[d, vn] = size(valD);
sv = alpha(alpha > 1e-9);
vio = size(sv,1);
obj = 0.5*w'*w;
fp = 0;
fn = 0;
tp = 0;
tn = 0;
for i = 1:vn
    sign = w'*valD(:,i)+b;
    zeta = 1 - valLb(i)*sign;
    if (zeta > 1e-9)
        obj = obj + C*zeta;
    end
    if (sign > 1e-9)
        if (valLb(i) == 1)
            tp = tp + 1;
        else
            fp = fp + 1;
        end
    end
    if (sign < -1e-9)
        if (valLb(i) == -1)
            tn = tn + 1;
        else
            fn = fn + 1;
        end
    end
end

acc = (vn-fp-fn)*1.0/vn;
fprintf('C = %f\n', C);
fprintf('Accuracy: %f\n',acc);
fprintf('Objective value of SVM: %f\n', obj);
fprintf('Number of support vectors: %d\n', vio);
fprintf('Confusion Matrix:\n');
fprintf('\t predicted +\tpredicted -\tsum\n');
fprintf('actual +%8d\t%7d \t%3d\n',tp, fn, sum(valLb==1));
fprintf('actual -%8d\t%7d \t%3d\n',fp, tn, sum(valLb==-1));
fprintf('sum     %8d\t%7d\n',tp+fp, fn+tn);

