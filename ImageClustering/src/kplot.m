function kplot()
iter = 10;
ssums = zeros(1,iter);
p1s = zeros(1,iter);
p2s = zeros(1,iter);
p3s = zeros(1,iter);
for i = 1:iter
    for j = 1:10
        [ssum, p1, p2, p3] = kmean(j);
        ssums(j) = ssums(j) + ssum;
        p1s(j) = p1s(j) + p1;
        p2s(j) = p2s(j) + p2;
        p3s(j) = p3s(j) + p3;
    end
end
ssums = ssums./iter;
p1s = p1s./iter;
p2s = p2s./iter;
p3s = p3s./iter;
xa = [1:iter];
assignin('base','ssums',ssums);
assignin('base','p1s',p1s);
assignin('base','p2s',p2s);
assignin('base','p3s',p3s);
f1 = figure
plot(xa, ssums);
xlabel('k');
ylabel('total within group sum of squares');
title('total within group sum of squares vs k');
saveas(f1,'ssums.png');
f2 = figure
plot(xa, p1s, xa, p2s, xa, p3s);
xlabel('k');
ylabel('fraction');
title('Ps vs k');
legend('p1','p2','p3');
saveas(f2,'ps.png');
end


function [ssum, p1, p2, p3] = kmean(k)
X = load('../hw5data/digit/digit.txt');
Y = load('../hw5data/digit/labels.txt');
X = X';
[fcent,rcent]=init(X,k);
[d,n] = size(X);
old_clst = zeros(1,n);
cent = rcent;
iter = 20;
for it = 1:iter
clst = assignClass(X, cent, k);
cent = centroid(X, clst, k);
if (clst == old_clst)
    %fprintf('number of iterations is %d \n',it);
    break;
end
old_clst = clst;
% assignin('base','clst',clst);
% assignin('base','cent',cent);
end

ssum = total_ssum(clst,X,cent,k);
[p1,p2,p3] = pair_count(X,clst,Y);
% fprintf('k = %d \n', k);
% fprintf('total with group sum of squares is %f \n',ssum);
% fprintf('p1=%f, p2=%f, p3=%f \n', p1, p2, p3);
end

function [p1,p2,p3] = pair_count(X,clst,Y)
Y = Y';
sets = unique(Y);
n_sets = size(sets,2);
num_p1 = 0;
num_p2 = 0;
t_p1 = 0;
t_p2 = 0;

classes = cell(1,n_sets);
for i = 1:n_sets
    class = (Y == sets(i));
    classes{i} = clst(class);
end

for i = 1:n_sets
    class_i = classes{i};
    sz_set = size(class_i,2);
    t_p1 = t_p1 + (sz_set-1)*sz_set*0.5;
    for j = 1:sz_set
        for k = j+1:sz_set
            if (class_i(j) == class_i(k))
                num_p1 = num_p1 + 1;
            end
        end
    end
end
p1 = num_p1*1.0/t_p1;

for i = 1:n_sets
    n_i = size(classes{i},2);
    for j = i+1:n_sets
        n_j = size(classes{j},2);
        t_p2 = t_p2 + n_i*n_j;
        for k = 1:n_i
            for l = 1:n_j
                if (classes{i}(k) ~= classes{j}(l))
                    num_p2 = num_p2 + 1;
                end
            end
        end
    end
end
p2 = num_p2*1.0/t_p2;

p3 = (p1+p2)*0.5;
end

function ssum = total_ssum(clst,X,cent,k)
[d,n] = size(X);
ss = zeros(1,k);
for i = 1:n
    label = clst(i);
    cent_i = cent(label);
    ss(label) = ss(label) + dist(X(:,i), X(:,cent_i));
end
ssum = sum(ss);
end

function cent = centroid(X, clst, k)
cent = zeros(1,k);
for i = 1:k
    grp = find(clst == i);
    assignin('base','clst',clst);
    assignin('base','i',i);
    assignin('base','grp',grp);
    cent(i) = findcenter(X, grp, clst);
end
end

function centid = findcenter(X, grp, clst)
n = size(grp,2);
assignin('base','n',n);
sqdist = zeros(1,n);
for i = 1:n
    for j = i+1:n
        delta = dist(X(:,grp(i)),X(:,grp(j)));
        sqdist(i) = sqdist(i) + delta;
        sqdist(j) = sqdist(j) + delta;
    end
    assignin('base','sqdist',sqdist);
end
[~,grpid] = min(sqdist);
centid = grp(grpid);
end

function cluster = assignClass(X, cent, k)
[d,n] = size(X);
cluster = zeros(1,n);
for i = 1:n
    dists = zeros(1, k);
    for j = 1:k
        dists(:,j) = dist(X(:,i), X(:,cent(:,j)));
    end
    [~, minC] = min(dists);
    cluster(i) = minC;
%     assignin('base','cluster',cluster);
%     assignin('base','k',k);
end
end

function d = dist(xi, xj)
dif = xi - xj;
d = dif'*dif;
end

function [firstK, rd] = init(X, k)
[d, n] = size(X);
if (k > n)
    fprintf('number of clusters is greater than the number of data.\n');
    return;
end
firstK = [1:k];
rd = randperm(n,k);
assignin('base','rd',rd);
assignin('base','k',k);
end


