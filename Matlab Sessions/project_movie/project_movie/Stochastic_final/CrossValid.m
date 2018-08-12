function [idx_validation, idx_train_new ] = CrossValid(idx_train, K_cross)

n_train = length(idx_train);

idx_cross = crossvalind('Kfold', n_train, K_cross);
for i = 1:K_cross
    idx_temp = find(idx_cross == i);
    idx_validation{i} = idx_train(idx_temp);
    
    idx_temp = find(idx_cross ~= i);
    idx_train_new{i} = idx_train(idx_temp);
end

end