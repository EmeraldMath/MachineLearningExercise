classdef KM
    
    methods (Static)
%         function main()         
%             X = load('../../hw5data/digit/digit.txt');
%             Y = load('../../hw5data/digit/labels.txt');
%             k = 1;
%             kmean(X,k);
%         end
        
        function cent = kmean(X,k)
            [fcent,rcent]=KM.init(X,k);
            [d,n] = size(X);
            old_clst = zeros(1,n);
            cent = rcent;
            iter = 20;
            for it = 1:iter
                clst = KM.assignClass(X, cent, k);
                cent = KM.centroid(X, clst, k);
                if (clst == old_clst)
                    fprintf('number of iterations is %d \n',it);
                    break;
                end
                old_clst = clst;
                % assignin('base','clst',clst);
                % assignin('base','cent',cent);
            end
        end
             
        function cent = centroid(X, clst, k)
            cent = zeros(1,k);
            for i = 1:k
                grp = find(clst == i);
                cent(i) = KM.findcenter(X, grp, clst);
            end
        end
        
        function centid = findcenter(X, grp, clst)
            n = size(grp,2);
            sqdist = zeros(1,n);
            for i = 1:n
                for j = i+1:n
                    delta = KM.dist(X(:,grp(i)),X(:,grp(j)));
                    sqdist(i) = sqdist(i) + delta;
                    sqdist(j) = sqdist(j) + delta;
                end
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
                    dists(:,j) = KM.dist(X(:,i), X(:,cent(:,j)));
                end
                [~, minC] = min(dists);
                cluster(i) = minC;
%                 if (i==40947)
%                     assignin('base','dists',dists);
%                 end
            end
%             assignin('base','cluster',cluster);
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
        end
        
    end
end