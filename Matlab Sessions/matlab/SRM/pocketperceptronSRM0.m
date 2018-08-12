function [bias, e_tr] = pocketperceptronSRM0(X, y, T)

[n , ~] = size(X);

perm = randperm(n);
X_rand = X(perm, :);
y_rand = y(perm);

bias = zeros(1, T+1);
k = 1;

ce(1) = 1/n * sum(sign(-bias(1, 1) + X_rand(:)) ~= y_rand);
    
stop = 0;

while (k <= T && not(stop))
    
    perm = randperm(n);
    X_rand = X(perm, :);
    y_rand = y(perm);
    
    i = 1;
    mistake = 0;
    
    while (i <= n && k <= T)
        
        signum = sign(-bias(1, k) + X_rand(i));
            
        if (y_rand(i) ~= signum)
       
            k = k + 1;
        
            bias(:, k) = bias(:, k-1) - 1/sqrt(k) * y_rand(i);
            ce(k) = 1/n * sum(sign(-bias(1, k) + X_rand(:)) ~= y_rand);
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

end