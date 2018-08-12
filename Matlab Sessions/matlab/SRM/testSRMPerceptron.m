function [] = testSRMPerceptron(n_test, n_rep)

T = 10000;
c = 0.1;
H = 4;

%Experimental scenarios
n = [10 100 10000];
num_s = length(n);

%Generate the test dataset
[X_t, y_t] = datageneration(n_test);
X0_t = X_t(:, 2);
X1_t = X_t;
X2_t = polyFeatures(X_t, 2);
X3_t = polyFeatures(X_t, 3);
X4_t = polyFeatures(X_t, 4);

ceH_tr = zeros(n_rep, H);
ceH_avg_tr = zeros(num_s, H);
ceH_std_dev_tr = zeros(num_s, H);
ceH = zeros(n_rep, H);

ce0_tr = zeros(n_rep, 1);
ce0_avg_tr = zeros(num_s, 1);
ce0_std_dev_tr = zeros(num_s, 1);
ce0 = zeros(n_rep, 1);

ce1_tr = zeros(n_rep, 1);
ce1_avg_tr = zeros(num_s, 1);
ce1_std_dev_tr = zeros(num_s, 1);
ce1 = zeros(n_rep, 1);

ce2_tr = zeros(n_rep, 1);
ce2_avg_tr = zeros(num_s, 1);
ce2_std_dev_tr = zeros(num_s, 1);
ce2 = zeros(n_rep, 1);

ce3_tr = zeros(n_rep, 1);
ce3_avg_tr = zeros(num_s, 1);
ce3_std_dev_tr = zeros(num_s, 1);
ce3 = zeros(n_rep, 1);

ce4_tr = zeros(n_rep, 1);
ce4_avg_tr = zeros(num_s, 1);
ce4_std_dev_tr = zeros(num_s, 1);
ce4 = zeros(n_rep, 1);

ce_tr = zeros(n_rep, 1);
ce_avg_tr = zeros(num_s, 1);
std_dev_tr = zeros(num_s, 1);

ce = zeros(n_rep, 1);
ce_avg = zeros(num_s, 1);
std_dev = zeros(num_s, 1);

H_calss = zeros(H+1,n_rep,num_s);

SRM = zeros(H+1,n_rep,num_s);

for j = 1:num_s
    
    for i = 1:n_rep

        fprintf('Experiment %d of %d. Repetition %d of %d.\n', j, num_s, i, n_rep);

        %Generate the training set
        [Train, y] = datageneration(n(j));
        X0 = Train(:, 2);
        X1 = Train;
        X2 = polyFeatures(Train, 2);
        X3 = polyFeatures(Train, 3);
        X4 = polyFeatures(Train, 4);
  
        %Run the Perceptron
        [bias0, ce0_tr(i)] = pocketperceptronSRM0(X0, y, T);

        
        [beta1, bias1, ce1_tr(i)] = pocketperceptronSRM1(X1, y, T, Train);
        [beta2, bias2, ce2_tr(i)] = pocketperceptronSRM1(X2, y, T, Train);
        [beta3, bias3, ce3_tr(i)] = pocketperceptronSRM1(X3, y, T, Train);
        [beta4, bias4, ce4_tr(i)] = pocketperceptronSRM1(X4, y, T, Train);
        
        
        %Penalties at delta =0.05
         p0 = c*sqrt(8/n(j) * log((4*(2*n(j)*exp(1))/1)^1/(1/5 * 0.05)));
         p1 = c*sqrt(8/n(j) * log((4*(2*n(j)*exp(1))/2)^2/(1/5 * 0.05)));
         p2 = c*sqrt(8/n(j) * log((4*(2*n(j)*exp(1))/3)^3/(1/5 * 0.05)));
         p3 = c*sqrt(8/n(j) * log((4*(2*n(j)*exp(1))/4)^4/(1/5 * 0.05)));
         p4 = c*sqrt(8/n(j) * log((4*(2*n(j)*exp(1))/5)^5/(1/5 * 0.05)));
         [value, idx] = min([ce0_tr(i)+p0 ce1_tr(i)+p1 ce2_tr(i)+p2 ce3_tr(i)+p3 ce4_tr(i)+p4]);
         
         H_calss(idx,i,j) = 1;
         SRM(:,i,j) = [ce0_tr(i)+p0 ce1_tr(i)+p1 ce2_tr(i)+p2 ce3_tr(i)+p3 ce4_tr(i)+p4]';
        
         if (idx == 1)
             
             ce_tr(i) = ce0_tr(i);
             
         elseif (idx == 2)
             
             ce_tr(i) = ce1_tr(i);
             
         elseif (idx == 3)
             
             ce_tr(i) = ce2_tr(i);
                 
         elseif (idx == 4)
             
             ce_tr(i) = ce3_tr(i);
             
         else 
             
             ce_tr(i) = ce4_tr(i);
             
         end

        signum = double(X0_t(:) > bias0);
        signum(find(signum==0)) = -1;
        ce0(i) = 1/n_test * sum(signum ~= y_t);        
        ce1(i) = 1/n_test * sum(sign(X1_t(:, end) - X1_t(:, 1:end-1)*beta1 - bias1*ones(n_test, 1)) ~= y_t);
        ce2(i) = 1/n_test * sum(sign(X2_t(:, end) - X2_t(:, 1:end-1)*beta2 - bias2*ones(n_test, 1)) ~= y_t);
        ce3(i) = 1/n_test * sum(sign(X3_t(:, end) - X3_t(:, 1:end-1)*beta3 - bias3*ones(n_test, 1)) ~= y_t);
        ce4(i) = 1/n_test * sum(sign(X4_t(:, end) - X4_t(:, 1:end-1)*beta4 - bias4*ones(n_test, 1)) ~= y_t);
        
        if (idx == 1)
            
            ce(i) = ce0(i);
            
        elseif (idx == 2)
            
             ce(i) = ce1(i);
            
        elseif (idx == 3)
            
            ce(i) = ce2(i);
            
        elseif (idx == 4)
            
            ce(i) = ce3(i);
            
        else
            
            ce(i) = ce4(i);
            
        end
        
    end
    
    %Error of H0 on the training set
    ce0_avg_tr(j) = mean(ce0_tr);
    ce0_std_dev_tr(j) = std(ce0_tr);
    SEM = std(ce0_tr)/sqrt(length(ce0_tr));          
    ts = tinv([0.05  0.95],length(ce0_tr)-1);     
    CI0_tr(j, :) = mean(ce0_tr) + ts*SEM;
    
    %Error of H0 on the test set
    ce0_avg(j) = mean(ce0);
    ce0_std_dev(j) = std(ce0);
    SEM = std(ce0)/sqrt(length(ce0));          
    ts = tinv([0.05  0.95],length(ce0)-1);     
    CI0(j, :) = mean(ce0) + ts*SEM;
    
    %Error of H1 on the training set
    ce1_avg_tr(j) = mean(ce1_tr);
    ce1_std_dev_tr(j) = std(ce1_tr);
    SEM = std(ce1_tr)/sqrt(length(ce1_tr));          
    ts = tinv([0.05  0.95],length(ce1_tr)-1);     
    CI1_tr(j, :) = mean(ce1_tr) + ts*SEM;
    
    %Error of H1 on the test set
    ce1_avg(j) = mean(ce1);
    ce1_std_dev(j) = std(ce1);
    SEM = std(ce1)/sqrt(length(ce1));          
    ts = tinv([0.05  0.95],length(ce1)-1);     
    CI1(j, :) = mean(ce1) + ts*SEM;
    
    %Error of H2 on the training set
    ce2_avg_tr(j) = mean(ce2_tr);
    ce2_std_dev_tr(j) = std(ce2_tr);
    SEM = std(ce2_tr)/sqrt(length(ce2_tr));          
    ts = tinv([0.05  0.95],length(ce2_tr)-1);     
    CI2_tr(j, :) = mean(ce2_tr) + ts*SEM;
    
    %Error of H2 on the test set
    ce2_avg(j) = mean(ce2);
    ce2_std_dev(j) = std(ce2);
    SEM = std(ce2)/sqrt(length(ce2));          
    ts = tinv([0.05  0.95],length(ce2)-1);     
    CI2(j, :) = mean(ce2) + ts*SEM;
    
    %Error of H3 on the training set
    ce3_avg_tr(j) = mean(ce3_tr);
    ce3_std_dev_tr(j) = std(ce3_tr);
    SEM = std(ce3_tr)/sqrt(length(ce3_tr));          
    ts = tinv([0.05  0.95],length(ce3_tr)-1);     
    CI3_tr(j, :) = mean(ce3_tr) + ts*SEM;
    
    %Error of H3 on the test set
    ce3_avg(j) = mean(ce3);
    ce3_std_dev(j) = std(ce3);
    SEM = std(ce3)/sqrt(length(ce3));          
    ts = tinv([0.05  0.95],length(ce3)-1);     
    CI3(j, :) = mean(ce3) + ts*SEM;
    
    %Error of H4 on the training set
    ce4_avg_tr(j) = mean(ce4_tr);
    ce4_std_dev_tr(j) = std(ce4_tr);
    SEM = std(ce4_tr)/sqrt(length(ce4_tr));          
    ts = tinv([0.05  0.95],length(ce4_tr)-1);     
    CI4_tr(j, :) = mean(ce4_tr) + ts*SEM;
    
    %Error of H4 on the test set
    ce4_avg(j) = mean(ce4);
    ce4_std_dev(j) = std(ce4);
    SEM = std(ce4)/sqrt(length(ce4));          
    ts = tinv([0.05  0.95],length(ce4)-1);     
    CI4(j, :) = mean(ce4) + ts*SEM;
    
    %Error of SRM on the training set
    ce_avg_tr(j) = mean(ce_tr);
    std_dev_tr(j) = std(ce_tr);
    SEM = std(ce_tr)/sqrt(length(ce_tr));          
    ts = tinv([0.05  0.95],length(ce_tr)-1);     
    CI_tr(j, :) = mean(ce_tr) + ts*SEM;          

    %Error of SRM on the test set
    ce_avg(j) = mean(ce);
    std_dev(j) = std(ce);    
    SEM = std(ce)/sqrt(length(ce));          
    ts = tinv([0.05  0.95],length(ce)-1);     
    CI(j, :) = mean(ce) + ts*SEM;
        
end
XX = 1:3;
figure(1);
hold on;
plot(ce_avg, 'ro-','LineWidth',2);
plot(ce_avg_tr,'bo-','LineWidth',2);

hold off;
title('SRM Learning curve: average');
ylabel('Error', 'FontSize', 24, 'FontWeight','bold');
xlabel('Training Set Size', 'FontSize', 24, 'FontWeight','bold');
set(gca,'XTick',[1 2 3]);
set(gca,'YTick',[0 0.05 0.1 0.15 0.2 0.25 0.3]);
set(gca,'XTickLabel',{'10', '100', '10000'}, 'FontSize', 24);
set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3'}, 'FontSize', 24);
legend('Test', 'Training')



figure(2);
hold on;
plot(ce0_avg, 'ro-','LineWidth',2);
plot(ce0_avg_tr, 'bo-','LineWidth',2);

hold off;
title('H0 Learning curve: average');
ylabel('Error', 'FontSize', 24, 'FontWeight','bold');
xlabel('Training Set Size', 'FontSize', 24, 'FontWeight','bold');
set(gca,'XTick',[1 2 3]);
set(gca,'YTick',[0 0.05 0.1 0.15 0.2 0.25 0.3]);
set(gca,'XTickLabel',{'10', '100', '10000'}, 'FontSize', 24);
set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3'}, 'FontSize', 24);
legend('Test', 'Training')


figure(3);
hold on;
plot(ce1_avg, 'ro-','LineWidth',2);
plot(ce1_avg_tr, 'bo-','LineWidth',2);

hold off;
title('H1 Learning curve: average');
ylabel('Error', 'FontSize', 24, 'FontWeight','bold');
xlabel('Training Set Size', 'FontSize', 24, 'FontWeight','bold');
set(gca,'XTick',[1 2 3]);
set(gca,'YTick',[0 0.05 0.1 0.15 0.2 0.25 0.3]);
set(gca,'XTickLabel',{'10', '100', '10000'}, 'FontSize', 24);
set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3'}, 'FontSize', 24);
legend('Test', 'Training')

figure(4);
hold on;
plot(ce2_avg, 'ro-','LineWidth',2);
plot(ce2_avg_tr, 'bo-','LineWidth',2);

hold off;
title('H2 Learning curve: average');
ylabel('Error', 'FontSize', 24, 'FontWeight','bold');
xlabel('Training Set Size', 'FontSize', 24, 'FontWeight','bold');
set(gca,'XTick',[1 2 3]);
set(gca,'YTick',[0 0.05 0.1 0.15 0.2 0.25 0.3]);
set(gca,'XTickLabel',{'10', '100', '10000'}, 'FontSize', 24);
set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3'}, 'FontSize', 24);
legend('Test', 'Training')

figure(5);
hold on;
plot(ce3_avg, 'ro-','LineWidth',2);
plot(ce3_avg_tr, 'bo-','LineWidth',2);

hold off;
title('H3 Learning curve: average');
ylabel('Error', 'FontSize', 24, 'FontWeight','bold');
xlabel('Training Set Size', 'FontSize', 24, 'FontWeight','bold');
set(gca,'XTick',[1 2 3]);
set(gca,'YTick',[0 0.05 0.1 0.15 0.2 0.25 0.3]);
set(gca,'XTickLabel',{'10', '100', '10000'}, 'FontSize', 24);
set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3'}, 'FontSize', 24);
legend('Test', 'Training')

figure(6);
hold on;
plot(ce4_avg, 'ro-','LineWidth',2);
plot(ce4_avg_tr, 'bo-','LineWidth',2);

hold off;
title('H4 Learning curve: average');
ylabel('Error', 'FontSize', 24, 'FontWeight','bold');
xlabel('Training Set Size', 'FontSize', 24, 'FontWeight','bold');
set(gca,'XTick',[1 2 3]);
set(gca,'YTick',[0 0.05 0.1 0.15 0.2 0.25 0.3]);
set(gca,'XTickLabel',{'10', '100', '10000'}, 'FontSize', 24);
set(gca,'YTickLabel',{'0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3'}, 'FontSize', 24);
legend('Test', 'Training')

figure(7);
hold on;
plot(mean(SRM(:,:,1),2), 'bo-','LineWidth',2);
plot(mean(SRM(:,:,2),2), 'ro-','LineWidth',2);
plot(mean(SRM(:,:,3),2), 'go-','LineWidth',2);
hold off;
title('Average augmented error, C=1');
ylabel('Error', 'FontSize', 24, 'FontWeight','bold');
xlabel('Training Set Size', 'FontSize', 24, 'FontWeight','bold');
set(gca,'XTick',[1 2 3 4 5]);
legend('n=10', 'n=100','n=10000');
set(gca,'XTickLabel',{'H0', 'H1', 'H2', 'H3', 'H4'}, 'FontSize', 24);
save('results_01')

end