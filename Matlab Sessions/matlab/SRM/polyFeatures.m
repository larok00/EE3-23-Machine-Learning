function [features] = polyFeatures(X, q)

n = length(X);
features = ones(n, 1);
x1 = X(:, 1);

for k = 1:q
    
    features(:, end + 1) = (x1.^(k));
            
end

features = [features(:, 2:end) X(:, 2)];

end