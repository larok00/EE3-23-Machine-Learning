function [E, E_cross, K, lambda, E_train] = stochastic_gradient_cross(Rating, idx_train, idx_test,lambda, K, itermax, eta, idx_cross, idx_train_cross, delta, K_cross, Every)

L_lambda = length(lambda);
L_k = length(K);
Error = zeros(L_lambda, L_k, K_cross);
E_train = zeros(L_lambda, L_k, K_cross);
for i = 1:L_lambda
    for j = 1:L_k
        for l = 1:K_cross
            
            idx_validation = idx_cross{l};
            
            idx_train_new = idx_train_cross{l};
            
            [~, ~, Error(i,j,l),E_train(i,j,l)] = stochastic_gradient(Rating, idx_train_new, idx_validation,lambda(i)/K(j), K(j), itermax, eta, delta, Every);
            [i j l]
        end
    end
end
Error = mean(Error,3);
Min = min(min(Error));
idx = find(Error==Min);
s = size(Error);
[I_opt, J_opt] = ind2sub(s, idx);
K = K(J_opt);
lambda = lambda(I_opt);
E_cross = Error;
[~, ~, E] = stochastic_gradient(Rating, idx_train, idx_test,lambda/K, K, itermax, eta, delta, Every);
end