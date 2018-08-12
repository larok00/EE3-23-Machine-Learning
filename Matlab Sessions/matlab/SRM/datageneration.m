function [X, y] = datageneration(n)
%Generate a dataset of n instances with d features distributed according
%to dist. Labels are generated accordingly a linear classifier. Returns also
%the classifier beta.
noise_level = 0.1;

%Generate the instances
x1 = rand(1, n) * 2.5;
x2 = -1 + rand(1, n) * 3;
X = [x1', x2'];

%Assign the labels
syms x;
f = @(x) [x.*(x-1).*(x-2)];       %symfun(x*(x-1)*(x-2), x);
y = sign(X(:, 2) - double(f(X(:, 1))));
y(find(y==0)) = -1;

q = binornd(n, noise_level);
y_p = randperm(length(y));
y_p = y_p(1:q);
y(y_p) = -1 * y(y_p);

%Plot the data if d = 2
plot_x = linspace(0, 2.5, 100)';
plot_y = double(f(plot_x));
   
% figure(1); 
% hold on;
% scatter(X(y==1, 1),X(y==1, 2),'bo');
% scatter(X(y==-1, 1),X(y==-1, 2),'rx');
% plot(plot_x, plot_y, 'k--');
% hold off
% xlabel('x1');  ylabel('x2');
% axis([0 2.5 -1 2]);
    
end