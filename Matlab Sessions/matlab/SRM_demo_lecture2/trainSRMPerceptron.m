function trainSRMPerceptron(noise,name,n_rep)

%name = 'lin';
%noise = 0.1;

%n_rep=100;
T = 100000; % max number of perceptrn updates
c = 0.1;  % scaler of the confidence bound


%Experimental scenarios
n = [10 50 100 1000];

num_s = length(n);
n_test = 18444; % 0.01 error with .95% confidence

% quadratic features
pF=@(x,y) [ x y x.*x x.*y y.*y ];
pF3=@(x,y) [ x y x.^2 x.*y y.^2  x.^3 x.^2.*y x.*y.^2 y.^3 x.^4 x.^3.*y x.^2.*y.^2 x.*y.^3 y.^4 ];
pFw2=@(x,y,w)x*w(1)+y*w(2);
pFw5=@(x,y,w)x*w(1)+y*w(2)+x.^2*w(3)+x.*y*w(4)+y.^2*w(5);
pFw14=@(x,y,w) x*w(1)+y*w(2)+x.^2*w(3)+x.*y*w(4)+y.^2*w(5)...
    + x.^3*w(6)+ x.^2.*y*w(7)+x.*y.^2*w(8)+ y.^3*w(9)...
    + x.^4*w(10)+ x.^3.*y*w(11)+ x.^2.*y.^2*w(12)+x.*y.^3*w(13)+y.^4*w(14);


%Generate the dataset

ce0_tr = zeros(n_rep, 1);
ce1_tr = zeros(n_rep, 1);
ce2_tr = zeros(n_rep, 1);
ce0_te = zeros(n_rep, 1);
ce1_te = zeros(n_rep, 1);
ce2_te = zeros(n_rep, 1);
ce_tr = zeros(n_rep,1);
ce_te = zeros(n_rep,1);
ind = zeros(n_rep,1);

for j = 1:num_s
    % penalties
    p0 = c*sqrt(8/n(j) * (3*log(2*n(j)*2.71/3)+log(12/0.05)));
    p1 = c*sqrt(8/n(j) * (6*log(2*n(j)*2.71/6)+log(12/0.05)));
    p2 = c*sqrt(8/n(j) * (15*log(2*n(j)*2.71/15)+log(12/0.05)));
    
    for i = 1:n_rep
        
        fprintf('Experiment %d of %d. Repetition %d of %d.\n', j, num_s, i, n_rep);
        
        %Generate training set
        [Train, y] = datageneration(n(j),noise,name);
        X0 = Train;
        X1 = pF(Train(:,1),Train(:,2));
        X2 = pF3(Train(:,1),Train(:,2));
        
        %Generate test set
        [Train_te, y_te] = datageneration(n_test,noise,name);
        X0_te = Train_te;
        X1_te = pF(Train_te(:,1),Train_te(:,2));
        X2_te = pF3(Train_te(:,1),Train_te(:,2));
        
        %Run the Perceptron
        [beta0, bias0] = pocketperceptronSRM1(X0, y, T);
        [beta1, bias1] = pocketperceptronSRM1(X1, y, T);
        [beta2, bias2] = pocketperceptronSRM1(X2, y, T);
        
        %Compute the errors
        
        ce0_tr(i) = 1/n(j) * sum(sign(X0*beta0 + bias0) ~= y);
        ce1_tr(i) = 1/n(j) * sum(sign(X1*beta1 + bias1) ~= y);
        ce2_tr(i) = 1/n(j) * sum(sign(X2*beta2 + bias2) ~= y);
        
        ce0_te(i) = 1/n_test * sum(sign(X0_te*beta0 + bias0) ~= y_te);
        ce1_te(i) = 1/n_test * sum(sign(X1_te*beta1 + bias1) ~= y_te);
        ce2_te(i) = 1/n_test * sum(sign(X2_te*beta2 + bias2) ~= y_te);
        
        %Penalties at delta =0.05
        [value, idx] = min([ce0_tr(i)+p0 ce1_tr(i)+p1 ce2_tr(i)+p2]);
        ind(i)=idx;
        if (idx == 1)
            ce_tr(i) = value-p0;
            ce_te(i) = ce0_te(i);
        elseif (idx == 2)
            ce_tr(i) = value-p1;
            ce_te(i) = ce1_te(i);
        elseif (idx == 3)
            ce_tr(i) = value-p2;
            ce_te(i) = ce2_te(i);
        end
        
    end
    close all;
    figure(1);
    hold on;
    [xx, yy] = meshgrid(-1:0.01:1, -1:0.01:1);
    pred0= pFw2(xx(:,:),yy(:,:),beta0)+bias0;
    pred1= pFw5(xx(:,:),yy(:,:),beta1)+bias1;
    pred2= pFw14(xx(:,:),yy(:,:),beta2)+bias2;
    mymap=[1 0.8 0.8; 0.5 1 1];colormap(mymap);
    contourf(xx,yy,pred0,[-1000 0 1000]);
    scatter(X0(y==1, 1),X0(y==1, 2),'bo');
    scatter(X0(y==-1, 1),X0(y==-1, 2),'rx');
    title(sprintf('%s n=%d noise=%.1f q=1',name,n(j),noise));
    print(sprintf('%s-%d-%d-1.jpg',name,n(j),10*noise),'-djpeg');
    %    pause;
    contourf(xx,yy,pred1,[-1000 0 1000]);
    scatter(X0(y==1, 1),X0(y==1, 2),'bo');
    scatter(X0(y==-1, 1),X0(y==-1, 2),'rx');
    title(sprintf('%s n=%d noise=%.1f q=2',name,n(j),noise));
    print(sprintf('%s-%d-%d-2.jpg',name,n(j),10*noise),'-djpeg');
    %    pause;
    contourf(xx,yy,pred2,[-1000 0 1000]);
    scatter(X0(y==1, 1),X0(y==1, 2),'bo');
    scatter(X0(y==-1, 1),X0(y==-1, 2),'rx');
    title(sprintf('%s n=%d noise=%.1f q=4',name,n(j),noise));
    print(sprintf('%s-%d-%d-4.jpg',name,n(j),10*noise),'-djpeg');
    %    pause;
    hold off;
    %write result to file
    filename=sprintf('res-%s-%d-%.1f-%d.txt',name,n(j),noise,n_rep);
    fileID = fopen(filename,'w');
    fprintf(fileID,'q=1 training: mean=%f var=%f penalty=%f selected=%f\n',mean(ce0_tr),var(ce0_tr),p0,sum(ind(ind(:)==1))/n_rep);
    fprintf(fileID,'q=1 test: mean=%f var=%f\n',mean(ce0_te),var(ce0_te));
    fprintf(fileID,'q=2 training: mean=%f var=%f penalty=%f selected=%f\n',mean(ce1_tr),var(ce1_tr),p1,sum(ind(ind(:)==2))/(2*n_rep));
    fprintf(fileID,'q=2 test: mean=%f var=%f\n',mean(ce1_te),var(ce1_te));
    fprintf(fileID,'q=4 training: mean=%f var=%f penalty=%f selected=%f\n',mean(ce2_tr),var(ce2_tr),p2,sum(ind(ind(:)==3))/(3*n_rep));
    fprintf(fileID,'q=4 test: mean=%f var=%f\n',mean(ce2_te),var(ce2_te));
    fprintf(fileID,'SRM training: mean=%f var=%f\n',mean(ce_tr),var(ce_tr));
    fprintf(fileID,'SRM test: mean=%f var=%f\n',mean(ce_te),var(ce_te));
    fclose(fileID);
end
end