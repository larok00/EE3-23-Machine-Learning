function [] = testH3Boundary(n)
%n is the number of training examples.
T = 100;

%Generate the training set
[X, y] = datageneration(n);

%Compute the features
[features] = polyFeatures(X, 3);

%Run a Pocket-PLA like algorithm and plot the decision boundary along the
%iterations
[beta, bias, e_tr, ce] = pocketperceptronSRM1(features, y, T, X);

fprintf('Training completed');
fprintf('Training Error of the learned classifier: %f.\n', e_tr);

%Generate the test set
[X_t, y_t] = datageneration(1000);
[features_t] = polyFeatures(X_t, 3);
ce_t = 1/1000 * sum((sign(features_t(:, end) - features_t(:, 1:end-1)*beta(1:end, 1) - bias*ones(1000, 1))) ~= y_t);
fprintf('Test Error of the learned classifier: %f.\n', ce_t);

end