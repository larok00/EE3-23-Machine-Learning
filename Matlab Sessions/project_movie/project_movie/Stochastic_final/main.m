close all
clear
clc

rng(9999999);
[Rating, idx_train, idx_test] = Rating2matrix();
[Gen] = Movies2matrix();
K_cross = 5; % Kfold cross validation 
[idx_cross, idx_train_cross] = CrossValid(idx_train, K_cross);

% Constant base predictors
[E_movie, E_user] = constant_predictor_cross(Rating, idx_cross, idx_test);

% Linear -> parameters
lambda = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 5e5 ,1e6, 5e6];
alpha = [0.1:.2:0.9]; % elastic net weight
% [E_l, E_lcross, opt] = Regression_cross(Rating, Gen, lambda, alpha, idx_test, idx_cross, K_cross); 

% Linear with new feature 
Gen_new = x2fx(Gen,'interaction'); % Genrates: Constant, linear, and interaction terms
Gen_new = Gen_new(:,2:end); % removing the constant column
% [E_l_new, E_lcross_new, opt_new] = Regression_cross(Rating, Gen_new, lambda, alpha, idx_test, idx_cross, K_cross);

%Stochastic -> parameters
K = [ 2 4 ]'; %[1 2 3 4 5 10 15 20 25]';
lambda = [ 1000]; % 0.0001 %[0.001, 0.005, 0.1, 0.5, 1, 5, 10, 50];
itermax = 10^8; % to stop the code in case of divergence 
Every = 5*10^6; 
delta = 1e-3;
eta = 0.01;
[E_s, E_scross, K_opt, lambda_opt, E_train] = stochastic_gradient_cross(Rating, idx_train, idx_test,lambda, K, itermax, eta, idx_cross, idx_train_cross, delta, K_cross, Every);

% save('results_new2.mat')
