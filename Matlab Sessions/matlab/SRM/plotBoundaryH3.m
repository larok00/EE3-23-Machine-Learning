function [] = plotBoundaryH3(X, y, beta)

plot_x = linspace(0, 2.5, 100)';
plot_y = poly(beta, plot_x);

figure(1); 
hold on;
scatter(X(y==1, 1),X(y==1, 2),'bo');
scatter(X(y==-1, 1),X(y==-1, 2),'rx');
plot(plot_x, plot_y, 'k--');
hold off
xlabel('x1');  ylabel('x2');
axis([0 2.5 -1 2]);

clear plot_y;

end

function f = poly(beta, x)
    
    disp('Compute Start');
    f = beta(1) + beta(2)*x + beta(3)*x.^2 + beta(4)*x.^3;
    disp('Compute end');
    
end