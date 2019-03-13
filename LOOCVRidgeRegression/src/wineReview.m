clear;
D = csvread('trainData.csv');
trLb = csvread('trainLabels.csv');
%lamda = [0.638, 0.64, 0.6425, 0.645];
%n = size(D,1);
lamda = [0.01, 0.1, 1, 10, 100, 1000];
itr = size(lamda,2);
rmsd_train = zeros(1,itr); 
rmsd_val = zeros(1, itr);
rmsd_loocv = zeros(1, itr);
loocv = zeros(1, itr);
for i = 1:itr
k = size(D,2);
X1 = D(:,2:k)';
y = trLb(:,2);
[w, b, obj, Errs] = train_ridgeReg(X1, y, lamda(i));
rmsd_train(i) = sqrt(Errs'*Errs/size(Errs,1));

vD = csvread('valData.csv');
vtrLb = csvread('valLabels.csv');
tmp = vD(:,2:k)';
vX1 = [tmp;ones(1,size(tmp,2))];
vy = vtrLb(:,2);
vErrs = vX1'*w-vy;
rmsd_val(i) = sqrt(vErrs'*vErrs/size(vErrs,1));

[loocv_w, loocv_b, loocv_obj, loocv_Errs] = ridgeReg(X1, y, lamda(i));
rmsd_loocv(i) = sqrt(loocv_Errs'*loocv_Errs/size(loocv_Errs,1));
loocv(i) = loocv_Errs'*loocv_Errs;
end
plotfig