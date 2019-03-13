classdef Xkernel
   methods (Static)
       
       function [trainK, testK] = main(trainD, testD)
           g = Xkernel.gamma(trainD);
           fprintf('gamma by mean is %f\n',g);
           [trainK, testK] = Xkernel.cmpExpX2Kernel(trainD, testD, g);
       end
       
       
       function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
           [n,d] = size(trainD);
           [tn,td] = size(testD);
           trainK = zeros(n, n+1);
           testK = zeros(tn, n+1);
           for i = 1:n
               trainK(i,1) = i;
               for j = 2:(n+1)
                   trainK(i,j) = exp(-Xkernel.chi(trainD(i,:), trainD(j-1,:))*(1.0/gamma));
               end
           end
           
           for i = 1:tn
               testK(i,1) = i;
               for j = 2:(n+1)
                   tmp = Xkernel.chi(testD(i,:), trainD(j-1,:));
                   testK(i,j) = exp(-tmp*(1.0/gamma));
               end
           end
           
       end
       
       
       function x = chi(u,v)
           epson = 1;
           [n,d] = size(u);
           x = 0;
           for i = 1:d
               x = x + (u(i)-v(i))*(u(i)-v(i))*1.0/(u(i)+v(i)+epson);
           end
       end
       
       function g = gamma(trainD)
           [n,d] = size(trainD);
           x = zeros(1,n*n);
           for i = 1:n
               for j = 1:n
                   x((i-1)*n+j) = Xkernel.chi(trainD(i,:), trainD(j,:));
               end
           end
           g = mean(x);
       end
       
   end
    
end