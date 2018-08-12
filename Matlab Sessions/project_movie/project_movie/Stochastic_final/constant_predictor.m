function [E_movie, E_user] = constant_predictor(Rate, idx_test, nr, mr)
Rate_original = Rate;
Rate(idx_test) = 0; % training set

C_movie = sum(Rate,1)./sum(Rate>0,1);
C_user = sum(Rate,2)./sum(Rate>0,2);

idx = find(isnan(C_movie));
C_movie(idx) = 2.5; %  Movies that are not included in the training set

%%% test 
L = length(idx_test);
error = zeros(L,2);

Rate = Rate_original;

for l = 1:L
    idx = idx_test(l);
    s = [nr, mr];
    [I, J] = ind2sub(s, idx);
    error(l,1) = (Rate(idx)-C_movie(J))^2;
    error(l,2) = (Rate(idx)-C_user(I))^2;
end

E_movie = mean(error(:,1));
E_user = mean(error(:,2));

end