function [E_movie, E_user] = constant_predictor_cross(Rating, idx_cross, idx_test)

K_cross = length(idx_cross);
[nr, mr] = size(Rating);
rating_train = Rating;
rating_train(idx_test) = 0; 

error = zeros(K_cross, 2);

for i = 1:K_cross
    idx_validation = idx_cross{i};
    [error(i,1), error(i,2)] = constant_predictor(rating_train, idx_validation, nr, mr);
end

if mean(error(:,1)) < mean(error(:,2))
    display('The best predictor: constant movie')
else
    display('The best predictor: constant user')
end

[E_movie, E_user] = constant_predictor(Rating, idx_test, nr, mr);


end
