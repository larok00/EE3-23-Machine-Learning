function [Error_final] = Regression_all(Rate, Gen, idx_test,lambda, method, alpha)
tiny = 1e-10;
Rate_original = Rate;
Rate(idx_test) = 0; % this removes the test data
K = size(Gen,2);
N = size(Rate,1);
U = zeros(N,K);

v_mean = zeros(K,N);
v_std = zeros(K,N);
r_mean = zeros(1,N);

for i = 1:N
    index = (Rate(i,:)>0);
    V = Gen(index,:)';
    r = Rate(i,index)';
    [n, m] =size(V);
    v_mean(:, i) = mean(V,2);
    v_std(:, i) = std(V')'+tiny;
    V_center = V-repmat(v_mean(:, i),1,m);
    V_norm = V_center./(repmat(v_std(:, i),1,m));
    V = V_norm;
    r_mean(i) = mean(r);
    r = r-r_mean(i);
    if strcmp(method,'ridge') 
        U(i,:) = ridge(r,V',lambda/K);
    elseif strcmp(method,'lasso')
        U(i,:) = lasso(V',r,'Lambda',lambda/K); 
    elseif strcmp(method,'Elasticnet')
        U(i,:) = lasso(V',r,'Lambda',lambda/K,'Alpha', alpha ); 
    end
end

%%% test
L = length(idx_test);
error = zeros(L,1);

Rate = Rate_original;
[nr, mr] = size(Rate);

for l = 1:L
    idx = idx_test(l);
    s = [nr, mr];
    [I, J] = ind2sub(s, idx);
    error(l) = (Rate(idx)-(U(I,:)*((Gen(J,:)' - v_mean(:, I))./v_std(:, I))+r_mean(I)))^2;
end

Error_final = sum(error)/L;

end