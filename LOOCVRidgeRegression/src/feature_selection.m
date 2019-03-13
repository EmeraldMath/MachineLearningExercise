clear;
D_l = csvread('trainData.csv');
trLb_l = csvread('trainLabels.csv');
lamda_l = 1;
%n = size(D,1);
k = size(D_l,2);
X1_l = D_l(:,2:k)';
y_l = trLb_l(:,2);
[w_l, b_l, obj_l, Errs_l] = train_ridgeReg(X1_l, y_l, lamda_l);

display(obj_l)
display(Errs_l'*Errs_l)

cut = w_l(1:size(w_l)-1);
display(cut'*cut)
para = abs(cut);
least = sort(para);
%min10 = least(1:10);
most = sort(para,'descend');
%max10 = most(1:10);
minindex = zeros(1,10);
maxindex = zeros(1,10);
for i = 1:10
    minindex(i) = find(para == least(i));
    maxindex(i) = find(para == most(i));
end

fileID = fopen('featureTypes.txt');
C = textscan(fileID,'%s','Delimiter',',');
fclose(fileID);

trival = strings(10,1);
vip = strings(10,1);
for i = 1:10
    trival(i) = C{1,1}{minindex(i),1};
    vip(i) = C{1,1}{maxindex(i),1};
end

tD = csvread('testData.csv');
tmp1 = tD(:,2:k)';
tX1 = [tmp1;ones(1,size(tmp1,2))];
predict = tX1'*w_l;
rank = [0:4999]';
output = [rank,predict];
cHeader = {'Id' 'Prediction'}; %dummy header
textHeader = strjoin(cHeader, ',');%write header to file
fid = fopen('predTestLabels.csv','w'); 
fprintf(fid,'%s\n',textHeader);
fclose(fid);
%write data to end of file
dlmwrite('predTestLabels.csv',output,'-append');
