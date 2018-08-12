function [E, error_all, Optimal] = Regression_cross(Rating, Gen, lambda, alpha ,idx_test, idx_cross, K_cross)

rating_training = Rating;
rating_training(idx_test) = 0;
L_lambda = length(lambda);
L_alpha = length(alpha);

error = zeros(K_cross, L_lambda,2);

for i = 1:K_cross
    for j = 1:L_lambda
        idx_validation = idx_cross{i};
        [error(i,j, 1)] = Regression_all(rating_training, Gen, idx_validation, lambda(j),'ridge',[]);
%         [error(i,j, 2)] = Regression_all(rating_training, Gen, idx_validation, lambda(j),'lasso',[]);
    end
end

% error_net = zeros(K_cross, L_lambda,L_alpha);
% for i = 1:K_cross
%     for j = 1:L_lambda
%         for k = 1:L_alpha
%             idx_validation = idx_cross{i};
% %             [error_net(i,j,k)] = Regression_all(rating_training, Gen, idx_validation, lambda(j),'Elasticnet',alpha(k));
%             [i j k]
%         end
%     end
% end

Optimal = zeros(1,4);
for i= 1:2
    [~, opt] = min(mean(error(:,:,i)));
    Optimal(i) = opt;
    
end
error_net = mean(error_net);
opt = min(min(min(error_net)));
idx = find(error_net == opt);
s = size(error_net);
[I, J, K] = ind2sub(s, idx);
Optimal(1,3:4) = [J(1) K(1)];


E = zeros(3,1);
E(1) = Regression_all(Rating, Gen, idx_test, lambda(Optimal(1)),'ridge',[]);
% E(2) = Regression_all(Rating, Gen, idx_test, lambda(Optimal(2)),'lasso',[]);
% E(3) = Regression_all(Rating, Gen, idx_test, lambda(Optimal(3)),'Elasticnet',alpha(Optimal(4)));

error_all{1} = mean(error);
error_all{2} = error_net;

end