function [beta, bias] = pocketperceptronSRM1(X, y, T)

[n , d] = size(X);
X = [ones(n, 1) , X];

perm = randperm(n);
X_rand = X(perm, :);
y_rand = y(perm);

beta = zeros(d+1, T+1);
k = 1;
ce(1) = 1/n * sum(sign(X_rand*beta(:, 1)) ~= y_rand);
stop = 0;

while (k <= T && not(stop))
    
    i = 1;
    mistake = 0;
    
    while (i <= n && k <= T)
        
        signum = sign(X_rand(i, :) * beta(:, k));
            
        if (y_rand(i) ~= signum)
       
            k = k + 1;
        
            beta(:, k) = beta(:, k-1) + 0.1*y_rand(i) * X_rand(i, :)';
            ce(k) = 1/n * sum(sign(X_rand*beta(:, k)) ~= y_rand);
            mistake = 1;
            
        end
        
        i = i + 1;
        
    end
    
    if (mistake == 0)
        
        stop = 1;
        
    end
    
end

[~, idx] = min(ce);
bias = beta(1, idx);
beta = beta(2:end, idx);

end