run ./vlfeat-0.9.21/toolbox/vl_setup;
dataDir = '../hw4data';
dataset='train';
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
% PosD = trD(:,1:2);
% negn = size(trD,2);
% NegD = trD(:,2:negn);
last = size(trLb,1);
index = find(trLb == -1, 1);
PosD = trD(:,1:(index-1));
NegD = trD(:,index:last);
[w, b, id, obj] = trainSVM(PosD, NegD);
loop = 10;
objs = zeros(1,loop);
AP = zeros(1,loop);
for iter = 1:loop
    B = hardexp(w, b, dataDir, 'train');
    NegD = [NegD(:,id),B];
    [w, b, id, obj] = trainSVM(PosD, NegD);
    objs(iter) = obj;
    HW4_Utils.genRsltFile(w, b, 'val', './hw4_3_4_3');
    %h = figure;
    [ap, prec, rec] = HW4_Utils.cmpAP('./hw4_3_4_3', 'val');
    AP(iter) = ap;
    %saveas(h,sprintf('./hw4_3_4_3/APinIteration%d.png',iter));
end
save('./hw4_3_4_3/objs.mat','objs');
save('./hw4_3_4_3/AP.mat','AP');
fig = figure;
plot([1:10], objs);
title('Objective Values for Each Iteration');
xlabel('Iteration');
ylabel('Objective Value');
saveas(fig,'./hw4_3_4_3/Obj.png');

fig = figure;
plot([1:10], AP);
title('AP for Each Iteration');
xlabel('Iteration');
ylabel('AP');
saveas(fig,'./hw4_3_4_3/AP.png');

function [w, b, id, obj] = trainSVM(PosD, NegD)
np = size(PosD,2);
nn = size(NegD,2);
X = [PosD, NegD];
Y = [ones(np,1);-ones(nn,1)];
C = 1; 
[H, f, A, b2, Aeq, beq, lb, ub] = setPar(X, Y, C);
[alpha, ~] = quadprog(H,f,A,b2,Aeq,beq,lb,ub);
alpha(alpha <= 1e-9) = 0;
[w, b] = primal(alpha, X, Y, C);

sv = all(alpha > 0, 2);
vio = find(sv == 1);
sv(1:np) = 0;
sz = size(sv,1);
%fprinf('sz %d\n',sz);
id = sv(np+1:sz);
obj = 0.5*(w'*w);
n_vio = size(vio,1);
for i = 1:n_vio
    obj = obj + C*(1-Y(vio(i))*(w'*X(:,vio(i))+b));
    %fprintf('zeta %f\n',(1-Y(vio(i))*(w'*X(:,vio(i))+b)));
end
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

function hep = hardexp(w, b, dataDir, dataset)
imFiles = ml_getFilesInDir(sprintf('%s/%sIms/', dataDir, dataset), 'jpg');
nIm = length(imFiles);            
rects = cell(1, nIm);
newF = cell(1, nIm);
num = 10;
for i=1:nIm
    load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, dataset), 'ubAnno');
    ubs = ubAnno{i};
    im = imread(imFiles{i});
    rects{i} = HW4_Utils.detect(im, w, b);
    for j=1:size(ubs,2)
        overlap = HW4_Utils.rectOverlap(rects{i}, ubs(:,j));                    
        rects{i} = rects{i}(:, overlap < 0.3);
         if isempty(rects{i})
             break;
         end
    end
    rects{i} = rects{i}(:,rects{i}(5,:) < 0);
    [~, id] = mink(rects{i}(5,:),num);
    hards = cat(2,rects{i}(1:4,id));
    nsz = size(hards,2);
    D_i = deal(cell(1, nsz));
    for j=1:nsz
        aRec = ceil(hards(:,j));
        if (aRec(4) > 360 || aRec(3) > 640)
            continue;
        end
        imReg = im(aRec(2):aRec(4), aRec(1):aRec(3),:);
        imReg = imresize(imReg, HW4_Utils.normImSz);
        D_i{j} = HW4_Utils.cmpFeat(rgb2gray(imReg));
    end 
    %each image
    newF{i} = cat(2, D_i{:});
end
hep = cat(2,newF{:});
hep = HW4_Utils.l2Norm(double(hep));
end