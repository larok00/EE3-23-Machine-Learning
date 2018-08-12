function [] = test()
A = rand(5,5);
A = (A + A')/2;

for i = 1:10000
    [V, D] = EIG(A);
    Error = EIG_reconstruct(A, V, D);    
end
end

function [V, D] = EIG(A)
[V, D] = eig(A);
end

function [error] = EIG_reconstruct(A, V, D)
error = sum(sum(A - V*D*V'));
end