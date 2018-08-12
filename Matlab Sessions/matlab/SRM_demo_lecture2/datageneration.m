function [X, y] = datageneration(n,noise_level,name)
%Generate a dataset of n instances with d features distributed according
%to dist. Labels are generated accordingly a linear classifier.

%Generate the instances
x1 = 2*rand(1, n)-1;
x2 = 2*rand(1, n)-1;
X = [x1', x2'];

%Assign the labels

if strcmp(name,'lin')
    y = 2*double(X(:, 1) + X(:, 2)<= 0)-1;
else
    y = 2*double(X(:, 1).^2 + X(:, 2).^2<= 0.49)-1;
end
    
%y(find(y==0)) = -1;

% add noise
noise=2*(rand(n,1)>=noise_level)-1;
y=y.*noise;
q = binornd(n,noise_level);
y_p = randperm(length(y));
y_p = y_p(1:q);
y(y_p) = -1 * y(y_p);

%Plot the data if d = 2
   
% figure(1); 
% hold on;
% scatter(X(y==1, 1),X(y==1, 2),'bo');
% scatter(X(y==-1, 1),X(y==-1, 2),'rx');
% xlabel('x1');  ylabel('x2');
% axis([-1 1 -1 1]);
    
end