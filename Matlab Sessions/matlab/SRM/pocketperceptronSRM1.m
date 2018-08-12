function [beta, bias, e_tr, ce] = pocketperceptronSRM1(X, y, T, Or_X)

[n , d] = size(X);
X = [ones(n, 1), X];

beta = zeros(d+1, T+1);
beta(end, :) = ones(1, T+1);
ce(1) = 1/n * sum((sign(X(:, end) - X(:, 1:end-1)*beta(1:end-1, 1))) ~= y);

stop = 0;
k = 1;

while (k < T && not(stop))

    %Permute the data
    perm = randperm(n);
    X_rand = X(perm, :);
    y_rand = y(perm);
    
    i = 1;
    mistake = 0;
    
    while (i <= n && k < T)
        
        signum = sign(X_rand(i, end) - X_rand(i, 1:end-1)*beta(1:end-1, k));
            
        if (y_rand(i) ~= signum)
       
            k = k + 1;
        
            beta(1:end-1, k) = beta(1:end-1, k-1) - 1/sqrt(k) * y_rand(i) * X_rand(i, 1:end-1)';
            ce(k) = 1/n * sum((sign(X_rand(:, end) - X_rand(:, 1:end-1)*beta(1:end-1, k))) ~= y_rand);
            mistake = 1;

        end
        
        i = i + 1;
        
    end
    
    if (mistake == 0)
        
        stop = 1;
        
    end
    
end

[e_tr, idx] = min(ce);
idx_test = find(ce == e_tr);
idx = idx_test(end);
bias = beta(1, idx);
beta = beta(2:end-1, idx);

end