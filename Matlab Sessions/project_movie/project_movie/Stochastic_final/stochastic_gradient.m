function [U, V, Error, E_train] = stochastic_gradient(Rating, idx_train, idx_test,lambda, K, itermax, eta, delta, Every)

%%%% normalization
Id = find(Rating>0);
R_mean = mean(Rating(Id));
R_std = std(Rating(Id));
Rating(Id) = (Rating(Id) - R_mean)./R_std;

%%%%%
[N, M] = size(Rating);
U = normrnd(0,1,N,K); %rand(N,K); % N: number of users
V = normrnd(0,1,M,K); %rand(M,K); % M: number of movies

% functions
dvjk = @(rij,ui,vj) (-2*ui*(rij-ui*vj')); %+lambda*vj);
duik = @(rij,ui,vj) (-2*vj*(rij-ui*vj')); %+lambda*ui);
lij = @(rij,ui,vj) ((rij-ui*vj')^2);

uind=zeros(N,1);
vind=zeros(M,1);
s = [N, M];
for iter=1:size(idx_train)
    [I, J] = ind2sub(s, idx_train(iter));
    uind(I)=uind(I)+1;
    vind(J)=vind(J)+1;
end
Udiag=ones(N,K)-eta*lambda*sqrt(uind)*ones(1,K);
Vdiag=ones(M,K)-eta*lambda*sqrt(vind)*ones(1,K);
uisqrt=sqrt(uind);
vjsqrt=sqrt(vind);
    
%%%% training
E_old = 1e100;
for outer = 1:itermax/Every
    idx = datasample(idx_train,Every);
    for iter  = 1:Every
        %U = Udiag.*U; %(1-eta*lambda)* U;
        %V = Vdiag.*V; %(1-eta*lambda)* V;
        
        %idx = datasample(idx_train,1);
        %s = [N, M];
        %[I, J] = ind2sub(s, idx(iter));
        I = rem(idx(iter)-1, N) + 1;
        J = (idx(iter) - I)/N + 1;
        update_v = dvjk(Rating(idx(iter)),U(I,:),V(J,:));
        update_u = duik(Rating(idx(iter)),U(I,:),V(J,:));
        U(I,:) = U(I,:)- eta*update_u/uisqrt(I);
        V(J,:) = V(J,:)- eta*update_v/vjsqrt(J);        
        
    end
    
    L = length(idx_train);
    error = zeros(L,1);
    s = [N, M];
    
    for l = 1:L
        idx = idx_train(l);
        [I, J] = ind2sub(s, idx);
        error(l) = lij(Rating(idx),U(I,:),V(J,:));
    end
    E_new = sum(error)/L;
    
    if abs(E_new - E_old)<= delta
        break 
    end
    outer
    E_old = E_new;    
    
end
E_train=E_new;
%%% test
L = length(idx_test);
error = zeros(L,1);

for l = 1:L
    idx = idx_test(l);
    [I, J] = ind2sub(s, idx);
    error(l) = lij(Rating(idx),U(I,:),V(J,:));
end
Error = sum(error)/L;

end